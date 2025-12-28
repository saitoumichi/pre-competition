import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import timm
import cv2
import socket
import torch.multiprocessing as mp
import hashlib

# ===========================
# 乱数固定（rankを混ぜる版）
# ===========================
def set_seed(seed=1234, rank=0):
    seed = seed + rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ===========================
# DDP 初期化（torchrun でも python 直実行でも動く版）
# ===========================

def ddp_setup(rank=None, local_rank=None, world_size=None, master_addr="127.0.0.1", master_port=29500):
    """DDP init.

    - torchrun で起動した場合: 環境変数(RANK/LOCAL_RANK/WORLD_SIZE等)をそのまま使う
    - python 直実行(spawn)の場合: rank/local_rank/world_size/master_* を引数で受け取り env:// で初期化する

    IMPORTANT:
      Windows では Docker Desktop / Kubernetes 等が残した MASTER_ADDR=kubernetes.docker.internal が
      環境変数に入っていることがあり、torchrunパスに誤判定すると警告が出る。
      そのため「引数が渡されている (= spawn)」場合は env の有無より優先する。
    """

    # -------------------------------------------------
    # python 直実行(spawn)パス（引数が渡されている場合は最優先）
    # -------------------------------------------------
    if rank is not None and local_rank is not None and world_size is not None:
        # 変な残留 env があっても確実に上書きする
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)

        if torch.cuda.is_available():
            torch.cuda.set_device(int(local_rank))

        backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        return int(rank), int(local_rank), int(world_size)

    # -------------------------------------------------
    # torchrun 起動パス（引数が無いときだけ env を採用）
    # -------------------------------------------------
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        # Windows環境で、環境変数のMASTER_ADDRが不正（例: kubernetes.docker.internal）になっていると
        # c10dが接続先として使ってしまい警告が出ることがある。
        # 単一ノード学習の想定では 127.0.0.1 に矯正して問題ない。
        if os.name == "nt":
            master_addr_env = os.environ.get("MASTER_ADDR", "")
            if ("kubernetes.docker.internal" in master_addr_env) or (master_addr_env.strip() == ""):
                os.environ["MASTER_ADDR"] = "127.0.0.1"

        backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        return rank, local_rank, world_size

    raise RuntimeError("ddp_setup: torchrunでもspawnでもない起動です。torchrunで起動するか、spawn時は rank/local_rank/world_size を渡してください")


def is_main_process(rank: int) -> bool:
    return rank == 0


def cleanup():
    dist.destroy_process_group()


# ===========================
# YOLOX: ROI cropper
# （まずは動く簡易版：416へ単純リサイズして推論→元画像へスケールで戻す）
# ===========================
def make_yolox_roi_cropper(yolox_model, postprocess_fn, test_size=(416, 416), conf=0.3, nms=0.65, pad_ratio=0.05):
    device = next(yolox_model.parameters()).device

    def cropper(pil_img: Image.Image) -> Image.Image:
        img_rgb = np.array(pil_img)  # HWC, RGB
        H, W = img_rgb.shape[:2]

        # PIL(RGB) -> BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 416へ単純リサイズ（簡易）
        resized = cv2.resize(img_bgr, test_size, interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)  # CHW
        x = torch.from_numpy(x).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = yolox_model(x)
            outputs = postprocess_fn(outputs, num_classes=1, conf_thre=conf, nms_thre=nms)

        if outputs[0] is None or outputs[0].shape[0] == 0:
            return pil_img

        det = outputs[0]
        scores = det[:, 4] * det[:, 5]  # obj * cls
        best = det[scores.argmax()]

        x1, y1, x2, y2 = best[:4].detach().cpu().numpy().tolist()

        # 416座標 -> 元画像座標へ戻す（簡易：リサイズのみ想定）
        sx = W / float(test_size[0])
        sy = H / float(test_size[1])
        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy

        # 余白
        bw = x2 - x1
        bh = y2 - y1
        x1 = int(max(0, x1 - pad_ratio * bw))
        y1 = int(max(0, y1 - pad_ratio * bh))
        x2 = int(min(W, x2 + pad_ratio * bw))
        y2 = int(min(H, y2 + pad_ratio * bh))

        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return pil_img

        return Image.fromarray(crop)

    return cropper


# ===========================
# ROI crop cache (disk) + Lazy YOLOX loader
# ===========================

def _roi_cache_key(path: str) -> str:
    """Cache key based on absolute path + mtime (so cache invalidates if file changes)."""
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = 0
    s = f"{os.path.abspath(path)}::{mtime}".encode("utf-8", errors="ignore")
    return hashlib.sha1(s).hexdigest()


class LazyYOLOXCropper:
    """Load YOLOX only when a cache miss happens (significantly reduces startup time on cached runs)."""

    def __init__(
        self,
        yolox_root: str,
        exp_file: str,
        ckpt_file: str,
        device: torch.device,
        test_size=(416, 416),
        conf: float = 0.3,
        nms: float = 0.65,
        pad_ratio: float = 0.05,
    ):
        self.yolox_root = yolox_root
        self.exp_file = exp_file
        self.ckpt_file = ckpt_file
        self.device = device
        self.test_size = test_size
        self.conf = conf
        self.nms = nms
        self.pad_ratio = pad_ratio

        self._cropper = None

    def _ensure_loaded(self):
        if self._cropper is not None:
            return

        # YOLOX を importできるようにパス追加
        if self.yolox_root not in sys.path:
            sys.path.insert(0, self.yolox_root)

        from yolox.exp import get_exp
        from yolox.utils import postprocess

        yexp = get_exp(self.exp_file, None)
        yexp.test_conf = self.conf
        yexp.nmsthre = self.nms
        yexp.test_size = self.test_size

        yolox_model = yexp.get_model().to(self.device)
        yolox_model.eval()

        ckpt = torch.load(self.ckpt_file, map_location=self.device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        yolox_model.load_state_dict(state, strict=False)

        self._cropper = make_yolox_roi_cropper(
            yolox_model=yolox_model,
            postprocess_fn=postprocess,
            test_size=self.test_size,
            conf=self.conf,
            nms=self.nms,
            pad_ratio=self.pad_ratio,
        )

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        self._ensure_loaded()
        return self._cropper(pil_img)


# ===========================
# Dataset クラス（YOLOX ROI対応）
# ===========================
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None, roi_detector=None, roi_cache_dir=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.roi_detector = roi_detector
        self.roi_cache_dir = roi_cache_dir
        if self.roi_cache_dir is not None:
            os.makedirs(self.roi_cache_dir, exist_ok=True)
        self.data = []
        self._prepare_data()

    def _prepare_data(self):
        for class_label in self.classes:
            class_path = os.path.join(self.root_dir, class_label)
            label_index = self.classes.index(class_label)

            if not os.path.isdir(class_path):
                continue

            for img_file in os.listdir(class_path):
                img_full_path = os.path.join(class_path, img_file)
                if os.path.isfile(img_full_path) and img_file.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
                ):
                    self.data.append((img_full_path, label_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        # ★追加：YOLOXでROI crop（disk cache対応）
        if self.roi_cache_dir is not None:
            cache_key = _roi_cache_key(img_path)
            cache_path = os.path.join(self.roi_cache_dir, f"{cache_key}.jpg")
            if os.path.exists(cache_path):
                try:
                    image = Image.open(cache_path).convert("RGB")
                except Exception:
                    pass
            elif self.roi_detector is not None:
                try:
                    image = self.roi_detector(image)
                    tmp_path = cache_path + f".{os.getpid()}.tmp"
                    try:
                        image.save(tmp_path, format="JPEG", quality=95)
                        os.replace(tmp_path, cache_path)
                    except Exception:
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass
                except Exception:
                    # 検出が壊れても学習を止めない（元画像で継続）
                    pass
        else:
            # cache無しの通常版
            if self.roi_detector is not None:
                try:
                    image = self.roi_detector(image)
                except Exception:
                    pass

        if self.transform:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None, roi_detector=None, roi_cache_dir=None):
        self.root_dir = root_dir
        self.transform = transform
        self.roi_detector = roi_detector
        self.roi_cache_dir = roi_cache_dir
        if self.roi_cache_dir is not None:
            os.makedirs(self.roi_cache_dir, exist_ok=True)
        self.image_files = [
            f for f in sorted(os.listdir(self.root_dir))
            if os.path.isfile(os.path.join(self.root_dir, f)) and
               f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fn = self.image_files[idx]
        path = os.path.join(self.root_dir, fn)
        image = Image.open(path).convert('RGB')

        # ★追加：YOLOXでROI crop（disk cache対応）
        if self.roi_cache_dir is not None:
            cache_key = _roi_cache_key(path)
            cache_path = os.path.join(self.roi_cache_dir, f"{cache_key}.jpg")
            if os.path.exists(cache_path):
                try:
                    image = Image.open(cache_path).convert("RGB")
                except Exception:
                    pass
            elif self.roi_detector is not None:
                try:
                    image = self.roi_detector(image)
                    tmp_path = cache_path + f".{os.getpid()}.tmp"
                    try:
                        image.save(tmp_path, format="JPEG", quality=95)
                        os.replace(tmp_path, cache_path)
                    except Exception:
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass
                except Exception:
                    pass
        else:
            if self.roi_detector is not None:
                try:
                    image = self.roi_detector(image)
                except Exception:
                    pass

        if self.transform:
            image = self.transform(image)
        return image, fn


# ===========================
# all_gather（可変長1次元tensorを集める）
# ===========================

def all_gather_1d_float_tensor(t: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    device = t.device

    local_n = torch.tensor([t.numel()], device=device, dtype=torch.long)
    sizes = [torch.zeros_like(local_n) for _ in range(world_size)]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes)

    padded = torch.zeros((max_n,), device=device, dtype=t.dtype)
    if t.numel() > 0:
        padded[:t.numel()] = t

    gathered = [torch.zeros((max_n,), device=device, dtype=t.dtype) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    out = []
    for g, n in zip(gathered, sizes):
        if n > 0:
            out.append(g[:n])
    if len(out) == 0:
        return torch.empty((0,), device=device, dtype=t.dtype)
    return torch.cat(out, dim=0)

# ===========================
# python直実行(spawn)用：空いているポートを探す
# ===========================

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

# ===========================
# threshold最適化（valでbest thresholdを探す）
# ===========================

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "acc_then_f1",
    prev_threshold: float = 0.5,
    tie_break: str = "closest_to_prev",
    search_mode: str = "local",
    search_radius: float = 0.10,
    num_points: int = 81,
):
    """y_true: {0,1} 1D array, y_prob: [0,1] 1D array
    Returns: (best_threshold, best_score)

    metric:
      - "acc_then_f1": まずAccuracy最大、その同点の中でF1最大、さらに同点なら tie_break に従う
      - "acc": Accuracy最大（同点は tie_break に従う）
      - "f1": F1最大
      - "youden": Youden index最大
      - "balanced_acc": Balanced accuracy最大
    """
    # --- threshold search grid ---
    # local: search in a narrow band around prev_threshold to reduce thr jumping (stabilizes F1)
    # global: full sweep (0.01..0.99)
    if search_mode == "local":
        pt = float(np.clip(prev_threshold, 0.01, 0.99))
        thresholds = np.clip(pt + np.linspace(-search_radius, search_radius, num_points), 0.01, 0.99)
        thresholds = np.concatenate([thresholds, np.array([0.5], dtype=np.float32)])
        thresholds = np.unique(np.round(thresholds.astype(np.float32), 4))
    else:
        thresholds = np.linspace(0.01, 0.99, 99).astype(np.float32)

    # 念のため型整形
    y_true = y_true.astype(np.int64)
    y_prob = y_prob.astype(np.float32)

    best_t = float(prev_threshold) if tie_break == "closest_to_prev" else 0.5

    def _tie_dist(tval: float) -> float:
        if tie_break == "closest_to_prev":
            return abs(float(tval) - float(prev_threshold))
        # default: closest_to_0.5
        return abs(float(tval) - 0.5)

    # 二段指標用（acc->f1）
    best_acc = -1.0
    best_f1 = -1.0

    # 単一指標用
    best_score = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int64)

        if metric == "acc_then_f1":
            acc = float((y_pred == y_true).mean())

            TP = np.sum((y_true == 1) & (y_pred == 1))
            FP = np.sum((y_true == 0) & (y_pred == 1))
            FN = np.sum((y_true == 1) & (y_pred == 0))

            prec = TP / (TP + FP) if (TP + FP) else 0.0
            rec = TP / (TP + FN) if (TP + FN) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

            # 1) Accuracy 最大
            # 2) Accuracy 同点なら F1 最大
            # 3) さらに同点なら tie_break に従う（例: prev_thresholdに近い）
            if (acc > best_acc) or \
               (acc == best_acc and f1 > best_f1) or \
               (acc == best_acc and f1 == best_f1 and _tie_dist(t) < _tie_dist(best_t)):
                best_acc = acc
                best_f1 = float(f1)
                best_t = float(t)

        else:
            if metric == "acc":
                score = float((y_pred == y_true).mean())
            elif metric == "f1":
                TP = np.sum((y_true == 1) & (y_pred == 1))
                FP = np.sum((y_true == 0) & (y_pred == 1))
                FN = np.sum((y_true == 1) & (y_pred == 0))

                prec = TP / (TP + FP) if (TP + FP) else 0.0
                rec = TP / (TP + FN) if (TP + FN) else 0.0
                score = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            elif metric == "youden":
                TP = np.sum((y_true == 1) & (y_pred == 1))
                FP = np.sum((y_true == 0) & (y_pred == 1))
                FN = np.sum((y_true == 1) & (y_pred == 0))
                TN = np.sum((y_true == 0) & (y_pred == 0))

                tpr = TP / (TP + FN) if (TP + FN) else 0.0
                fpr = FP / (FP + TN) if (FP + TN) else 0.0
                score = tpr - fpr
            elif metric == "balanced_acc":
                TP = np.sum((y_true == 1) & (y_pred == 1))
                FP = np.sum((y_true == 0) & (y_pred == 1))
                FN = np.sum((y_true == 1) & (y_pred == 0))
                TN = np.sum((y_true == 0) & (y_pred == 0))

                tpr = TP / (TP + FN) if (TP + FN) else 0.0
                tnr = TN / (TN + FP) if (TN + FP) else 0.0
                score = 0.5 * (tpr + tnr)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            # 同点なら tie_break に従う
            if (score > best_score) or (score == best_score and _tie_dist(t) < _tie_dist(best_t)):
                best_score = float(score)
                best_t = float(t)

    if metric == "acc_then_f1":
        # best_score には primary の Accuracy を返す（ログ用/整合性）
        return best_t, float(best_acc)
    return best_t, float(best_score)


def main(local_rank=None, world_size=None, master_addr="127.0.0.1", master_port=None):
    # torchrun なら env から、python直実行(spawn)なら引数から DDP を初期化
    if local_rank is None:
        rank, local_rank, world_size = ddp_setup()
    else:
        if master_port is None:
            raise RuntimeError("spawn起動時は master_port が必要です")
        rank = int(local_rank)  # 単一ノード想定なので rank=local_rank
        rank, local_rank, world_size = ddp_setup(
            rank=rank,
            local_rank=int(local_rank),
            world_size=int(world_size),
            master_addr=master_addr,
            master_port=int(master_port),
        )

    set_seed(1234, rank=rank)

    device = torch.device(f"cuda:{local_rank}")

    train_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer_mk2\train"
    val_dir   = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\valid"
    test_dir  = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\test"

    log_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_kazma\log_GPU2"
    writer = SummaryWriter(log_dir=log_dir) if is_main_process(rank) else None

    output_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_michi\GPU2"

    classes = ["0", "1"]
    # ===========================
    # Tunable config (change here)
    # ===========================
    # ConvNeXt: tiny/small を切り替え（まずは tiny 推奨：軽くて過学習しにくい）
    # 例: "convnext_tiny.fb_in22k_ft_in1k_384" or "convnext_small.fb_in22k_ft_in1k_384"
    MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k_384"

    # 640x640 学習/推論設定
    IMG_SIZE = 640            # input resolution
    VAL_RESIZE = 704          # Resize -> CenterCrop(640) 用（少し大きめ）

    # 640はメモリを食うので小さめで
    BATCH_SIZE = 2
    NUM_WORKERS = 0           # Windowsで不安定なら0のまま。余裕があれば 2〜4。

    # 学習するなら 0 ではなく >0 にする（推論だけなら 0 のままでOK）
    NUM_EPOCHS = 25

    # 高解像度は不安定になりやすいので少し控えめに
    LR = 3e-5
    WEIGHT_DECAY = 0.05       # AdamWの定番

    # 実効バッチを稼ぐ（BATCH_SIZE=2 でも 2*4=8 相当）
    ACCUM_STEPS = 4

    # AMP (mixed precision) で速度/メモリ改善
    USE_AMP = torch.cuda.is_available()

    # augmentation strength
    ROT_DEG = 10
    # 640は情報量が多いので、強い切り取りで病変が消えやすい → 少し弱める
    RRC_SCALE = (0.90, 1.00)
    RRC_RATIO = (0.95, 1.05)

    # early stopping (val_acc(opt) 기준)
    EARLY_STOPPING = True
    ES_PATIENCE = 5
    ES_MIN_DELTA = 1e-4

    # ===========================
    # ★追加：重み保存先（bestもここに保存）
    # ===========================
    weight_dir = os.path.join(output_dir, "Weight")
    if is_main_process(rank):
        os.makedirs(weight_dir, exist_ok=True)
    dist.barrier()

    def save_ckpt(name: str, epoch_i: int, thr: float, metrics: dict):
        ckpt = {
            "epoch": int(epoch_i),
            "threshold": float(thr),
            "metrics": {k: float(v) for k, v in metrics.items()},
            "state_dict": model.module.state_dict(),
        }
        torch.save(ckpt, os.path.join(weight_dir, f"best_{name}.pth"))

    best = {
        "acc": (-1.0, -1),
        "loss": (1e18, -1),
        "auc": (-1.0, -1),
    }

    # ===========================
    # ★変更：YOLOXは「必要になった時だけ」ロード（cache miss時）
    # さらにROI crop結果をディスクへキャッシュして、次回以降の起動/学習を高速化
    # ===========================
    yolox_root = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\YOLOX"

    exp_file = os.path.join(yolox_root, r"exps\custom\yolox_nano_breast_roi_416.py")
    ckpt_file = os.path.join(yolox_root, r"YOLOX_outputs\yolox_base\latest_ckpt.pth")

    roi_cache_dir = os.path.join(output_dir, "roi_cache_yolox416_pad010")
    if is_main_process(rank):
        os.makedirs(roi_cache_dir, exist_ok=True)
    dist.barrier()

    roi_cropper = LazyYOLOXCropper(
        yolox_root=yolox_root,
        exp_file=exp_file,
        ckpt_file=ckpt_file,
        device=device,
        test_size=(416, 416),
        conf=0.3,
        nms=0.65,
        pad_ratio=0.10,
    )

    # ===========================
    # Transform
    # ===========================
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(
            size=IMG_SIZE,
            scale=RRC_SCALE,
            ratio=RRC_RATIO,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(ROT_DEG),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(VAL_RESIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ★変更：roi_detector=roi_cropper, roi_cache_dir=roi_cache_dir を渡す
    train_dataset = BreastCancerDataset(train_dir, classes, train_transforms, roi_detector=roi_cropper, roi_cache_dir=roi_cache_dir)
    val_dataset   = BreastCancerDataset(val_dir, classes, val_test_transforms, roi_detector=roi_cropper, roi_cache_dir=roi_cache_dir)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler   = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1).to(device)
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    else:
        model = DDP(model)

    pos_weight_value = 1.4160455940377028
    pos_weight = torch.tensor([pos_weight_value], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, NUM_EPOCHS), eta_min=LR * 0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    num_epochs = NUM_EPOCHS
    # early stopping state (rank0 only)
    best_acc_es = -1.0
    no_improve = 0
    # ===========================
    # ★追加：valで動的に最適化して更新していく閾値（初期値）
    # ===========================
    threshold = 0.4

    # ===========================
    # ★追加：bestモデル保存（rank0のみ）
    # ===========================

    if writer is not None:  #convnext_base.fb_in22k_ft_in1k_384
        writer.add_text("config", f"""
model={MODEL_NAME}
batch_size={BATCH_SIZE}
lr={LR}
weight_decay={WEIGHT_DECAY}
epochs={num_epochs}
img_size={IMG_SIZE}
pos_weight={pos_weight_value}
threshold=dynamic (init=0.4)
log_dir={log_dir}
world_size={world_size}
roi=yolox_nano_breast_roi_416 (416)
""")

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        global_step = epoch * len(train_loader)

        epoch_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Train", leave=False) if is_main_process(rank) else train_loader

        for inputs, labels in epoch_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True).view(-1, 1)

            # gradient accumulation
            if (global_step % ACCUM_STEPS) == 0:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(inputs).view(-1, 1)
                loss = criterion(outputs, labels)
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if ((global_step + 1) % ACCUM_STEPS) == 0:
                scaler.step(optimizer)
                scaler.update()

            # loss は /ACCUM_STEPS 済みなので、ログ用に元スケールへ戻す
            running_loss += (loss.item() * ACCUM_STEPS) * inputs.size(0)

            if writer is not None:
                writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            global_step += 1

            preds = (torch.sigmoid(outputs) >= threshold).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        running_loss_t = torch.tensor([running_loss], device=device)
        correct_t = torch.tensor([correct], device=device, dtype=torch.long)
        total_t = torch.tensor([total], device=device, dtype=torch.long)

        dist.all_reduce(running_loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)

        epoch_loss = running_loss_t.item() / max(1, len(train_dataset))
        epoch_acc = correct_t.item() / max(1, total_t.item())

        # -------------------------------
        # Validation
        # -------------------------------
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0

        local_probs = []
        local_labels = []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Val", leave=False) if is_main_process(rank) else val_loader

            for inputs, labels in val_bar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True).view(-1, 1)

                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    outputs = model(inputs).view(-1, 1)
                    loss = criterion(outputs, labels)

                # --- debug: investigate abnormal val_loss spikes (rank0 only) ---
                if is_main_process(rank):
                    finite_loss = bool(torch.isfinite(loss).all().item())
                    finite_out = bool(torch.isfinite(outputs).all().item())
                    max_abs_logit = float(outputs.detach().abs().max().item())
                    if (not finite_loss) or (not finite_out) or (max_abs_logit > 100.0):
                        print(
                            "[VAL-DEBUG]",
                            f"loss={float(loss.detach().item())}",
                            f"logit_min={float(outputs.detach().min().item())}",
                            f"logit_max={float(outputs.detach().max().item())}",
                            f"max_abs_logit={max_abs_logit}",
                            f"label_unique={labels.detach().unique().cpu().numpy()}",
                        )

                val_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs >= threshold).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                local_probs.append(probs.view(-1))
                local_labels.append(labels.view(-1))

        val_loss_t = torch.tensor([val_loss], device=device)
        val_correct_t = torch.tensor([val_correct], device=device, dtype=torch.long)
        val_total_t = torch.tensor([val_total], device=device, dtype=torch.long)

        dist.all_reduce(val_loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_total_t, op=dist.ReduceOp.SUM)

        val_epoch_loss = val_loss_t.item() / max(1, len(val_dataset))
        val_epoch_acc = val_correct_t.item() / max(1, val_total_t.item())

        local_probs_t = torch.cat(local_probs, dim=0) if len(local_probs) else torch.empty((0,), device=device)
        local_labels_t = torch.cat(local_labels, dim=0) if len(local_labels) else torch.empty((0,), device=device)

        gathered_probs = all_gather_1d_float_tensor(local_probs_t)
        gathered_labels = all_gather_1d_float_tensor(local_labels_t)

        # -------------------------------
        # ★追加：valの確率から best threshold を探索し、全rankへ共有
        # -------------------------------
        prev_threshold = float(threshold)
        best_t = threshold
        best_score = float("nan")

        if is_main_process(rank):
            y_prob = gathered_probs.detach().cpu().numpy()
            y_true = gathered_labels.detach().cpu().numpy()

            # コンペの評価指標が Accuracy なので、val上のAccuracyが最大になる閾値を探す
            best_t, best_score = find_best_threshold(
                y_true,
                y_prob,
                metric="acc_then_f1",
                prev_threshold=prev_threshold,
                tie_break="closest_to_prev",
                search_mode="local",
                search_radius=0.10,
                num_points=81,
            )

        t_tensor = torch.tensor([best_t], device=device, dtype=torch.float32)
        dist.broadcast(t_tensor, src=0)
        threshold = float(t_tensor.item())

        if is_main_process(rank):
            # best threshold を使ってval指標を再計算
            y_pred = (y_prob >= threshold).astype(np.int64)

            acc_opt = accuracy_score(y_true, y_pred)
            pred_pos_rate = float((y_pred == 1).mean())
            true_pos_rate = float((y_true == 1).mean())

            TP = np.sum((y_true == 1) & (y_pred == 1))
            FP = np.sum((y_true == 0) & (y_pred == 1))
            FN = np.sum((y_true == 1) & (y_pred == 0))
            prec = TP / (TP + FP) if (TP + FP) else 0.0
            rec = TP / (TP + FN) if (TP + FN) else 0.0
            f1_opt = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
            cm = confusion_matrix(y_true, y_pred)

            metrics_now = {
                "acc": float(acc_opt),
                "f1": float(f1_opt),
                "loss": float(val_epoch_loss),
                "auc": float(auc) if auc == auc else float("nan"),
                "thr": float(threshold),
                "pred_pos_rate": float(pred_pos_rate),
                "true_pos_rate": float(true_pos_rate),
                # find_best_threshold(metric="acc_then_f1") は best_score=best_acc を返す
                "thr_primary": float(best_score) if best_score == best_score else float("nan"),
            }

            # best acc
            if acc_opt > best["acc"][0]:
                best["acc"] = (float(acc_opt), epoch + 1)
                save_ckpt("acc", epoch + 1, threshold, metrics_now)
                print(f"[BEST-ACC] epoch={epoch+1} acc={acc_opt:.6f} -> saved best_acc.pth")

            # best loss (min)
            if val_epoch_loss < best["loss"][0]:
                best["loss"] = (float(val_epoch_loss), epoch + 1)
                save_ckpt("loss", epoch + 1, threshold, metrics_now)
                print(f"[BEST-LOSS] epoch={epoch+1} loss={val_epoch_loss:.6f} -> saved best_loss.pth")

            # best auc
            if auc == auc and auc > best["auc"][0]:
                best["auc"] = (float(auc), epoch + 1)
                save_ckpt("auc", epoch + 1, threshold, metrics_now)
                print(f"[BEST-AUC] epoch={epoch+1} auc={auc:.6f} -> saved best_auc.pth")

            if cm.size == 4:
                TN, FP, FN, TP = cm.ravel()
                sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0.0
                specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0
            else:
                sensitivity, specificity = float("nan"), float("nan")

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"train_loss={epoch_loss:.6f} train_acc={epoch_acc:.6f} | "
                  f"val_loss={val_epoch_loss:.6f} val_acc(opt)={acc_opt:.6f} val_f1(opt)={f1_opt:.4f} | "
                  f"thr={threshold:.2f} (prev={prev_threshold:.2f}) thr_score={best_score:.4f} | "
                  f"pred_pos_rate={pred_pos_rate:.3f} true_pos_rate={true_pos_rate:.3f} | "
                  f"AUC={auc} Sens={sensitivity} Spec={specificity}")

            if writer is not None:
                writer.add_scalars("Loss", {"train": epoch_loss, "val": val_epoch_loss}, epoch)
                writer.add_scalars("Accuracy", {"train": epoch_acc, "val_opt": acc_opt}, epoch)
                writer.add_scalar("Val/F1_opt", f1_opt, epoch)
                writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
                writer.add_scalar("Val/AUC", auc if auc == auc else 0.0, epoch)
                writer.add_scalar("Val/Sensitivity", sensitivity if sensitivity == sensitivity else 0.0, epoch)
                writer.add_scalar("Val/Specificity", specificity if specificity == specificity else 0.0, epoch)
                writer.add_scalar("Val/BestThreshold", threshold, epoch)
                writer.add_scalar("Val/BestThresholdScore", best_score if best_score == best_score else 0.0, epoch)
                writer.add_scalar("Val/PredPosRate", pred_pos_rate, epoch)
                writer.add_scalar("Val/TruePosRate", true_pos_rate, epoch)

                if cm.size == 4:
                    fig = plt.figure()
                    plt.imshow(cm, interpolation="nearest")
                    plt.title("Confusion Matrix (val) @ best threshold")
                    plt.colorbar()
                    tick_marks = np.arange(2)
                    plt.xticks(tick_marks, ["0", "1"])
                    plt.yticks(tick_marks, ["0", "1"])
                    plt.xlabel("Pred")
                    plt.ylabel("True")
                    for i in range(2):
                        for j in range(2):
                            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
                    writer.add_figure("Val/ConfusionMatrix", fig, epoch)
                    plt.close(fig)

        # -------------------------------
        # Early stopping decision (rank0 computes, then broadcast)
        # -------------------------------
        stop_tensor = torch.zeros((1,), device=device, dtype=torch.int32)

        if EARLY_STOPPING and is_main_process(rank):
            # acc_opt は rank0 のみで計算されるので、ここでのみ参照する
            if acc_opt > best_acc_es + ES_MIN_DELTA:
                best_acc_es = float(acc_opt)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= ES_PATIENCE:
                stop_tensor[0] = 1

        # broadcast early-stop flag (all ranks)
        dist.broadcast(stop_tensor, src=0)
        should_stop = bool(int(stop_tensor.item()) == 1)

        # scheduler step only if continuing
        if writer is not None and is_main_process(rank):
            writer.add_scalar("LR/Scheduler", scheduler.get_last_lr()[0], epoch)

        if not should_stop:
            scheduler.step()

        dist.barrier()

        if should_stop:
            if is_main_process(rank):
                print(f"[EARLY-STOP] Stop at epoch={epoch+1} (best_acc={best_acc_es:.6f}, patience={ES_PATIENCE})")
            break

    if writer is not None:
        writer.close()

    # ===========================
    # 重み保存（rank0のみ）
    # ===========================

    dist.barrier()
    if is_main_process(rank):
        os.makedirs(weight_dir, exist_ok=True)
        torch.save(model.module.state_dict(), os.path.join(weight_dir, "breast_cancer_model_ddp.pth"))
        print(f"Saved weights to: {os.path.join(weight_dir, 'breast_cancer_model_ddp.pth')}")

    dist.barrier()

    # ===========================
    # TTA (Test-Time Augmentation) + Checkpoint Ensemble
    # ===========================
    # 推論だけ強化してスコアを上げやすい最短ルート。
    # まずは hflip のみON推奨（vflip/rot90はデータ次第で効いたり悪化したりするので任意）。
    TTA_HFLIP = True
    TTA_VFLIP = False
    TTA_ROT90 = True  # 90度回転（ONにするなら 90/270 の平均を取る）

    # アンサンブル対象（存在するものだけ使う）
    ENSEMBLE_CKPTS = ["best_acc.pth", "best_auc.pth", "best_loss.pth"]

    # ===========================
    # Submission threshold policy
    # ===========================
    # Public/hidden test distribution can differ from val, so a val-optimized threshold may hurt LB.
    # Start with a fixed 0.5 threshold for submission; you can sweep thresholds later using the saved prob file.
    USE_CKPT_THRESHOLD = False   # if True, load threshold from best_acc.pth
    FIXED_SUBMIT_THRESHOLD = 0.25

    def infer_probs_with_tta(model_ddp, images: torch.Tensor) -> torch.Tensor:
        """Return (B,) probs averaged over TTA."""
        probs_list = []

        # original
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            probs_list.append(torch.sigmoid(model_ddp(images).view(-1)))

        # hflip
        if TTA_HFLIP:
            images_h = torch.flip(images, dims=[3])  # NCHW の W 方向（左右反転）
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                probs_list.append(torch.sigmoid(model_ddp(images_h).view(-1)))

        # vflip
        if TTA_VFLIP:
            images_v = torch.flip(images, dims=[2])  # NCHW の H 方向（上下反転）
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                probs_list.append(torch.sigmoid(model_ddp(images_v).view(-1)))

        # hvflip
        if TTA_HFLIP and TTA_VFLIP:
            images_hv = torch.flip(images, dims=[2, 3])
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                probs_list.append(torch.sigmoid(model_ddp(images_hv).view(-1)))

        # rot90 / rot270
        if TTA_ROT90:
            images_r90 = torch.rot90(images, k=1, dims=[2, 3])
            images_r270 = torch.rot90(images, k=3, dims=[2, 3])
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                probs_list.append(torch.sigmoid(model_ddp(images_r90).view(-1)))
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                probs_list.append(torch.sigmoid(model_ddp(images_r270).view(-1)))

        probs = torch.stack(probs_list, dim=0).mean(dim=0)
        return probs

    # ===========================
    # テスト推論（ROI crop → 分類）
    # ===========================
    dist.barrier()
    torch.cuda.empty_cache()

    # ★変更：roi_detector=roi_cropper, roi_cache_dir=roi_cache_dir を渡す
    test_dataset = TestDataset(test_dir, val_test_transforms, roi_detector=roi_cropper, roi_cache_dir=roi_cache_dir)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # --- threshold policy (submission) ---
    dist.barrier()
    best_acc_path = os.path.join(weight_dir, "best_acc.pth")

    if USE_CKPT_THRESHOLD and os.path.exists(best_acc_path):
        ckpt = torch.load(best_acc_path, map_location=device)
        threshold = float(ckpt.get("threshold", threshold))
    else:
        threshold = float(FIXED_SUBMIT_THRESHOLD)

    dist.barrier()

    # --- ensemble ckpt paths（存在するものだけ） ---
    ckpt_paths = []
    for name in ENSEMBLE_CKPTS:
        p = os.path.join(weight_dir, name)
        if os.path.exists(p):
            ckpt_paths.append(p)

    # 何も見つからない場合のフォールバック（今の重みで推論）
    if len(ckpt_paths) == 0 and os.path.exists(best_acc_path):
        ckpt_paths = [best_acc_path]

    model.eval()

    # 画像ごとに「確率の合計」と「加算回数」を持って最後に平均 → 閾値判定
    local_prob_sum = {}
    local_prob_cnt = {}

    with torch.no_grad():
        # checkpoint ensemble
        for ckpt_path in ckpt_paths:
            # 各rankで同じckptをロード
            ckpt = torch.load(ckpt_path, map_location=device)
            model.module.load_state_dict(ckpt["state_dict"], strict=True)
            model.eval()

            # samplerの順序を固定（repeatしても同じ画像割当になるように）
            try:
                test_sampler.set_epoch(0)
            except Exception:
                pass

            test_bar = tqdm(test_loader, desc=f"Test Prediction ({os.path.basename(ckpt_path)})", leave=False) if is_main_process(rank) else test_loader
            for images, filenames in test_bar:
                images = images.to(device, non_blocking=True)

                probs = infer_probs_with_tta(model, images)
                probs_cpu = probs.detach().cpu().numpy().tolist()

                for fn, pr in zip(filenames, probs_cpu):
                    image_id = os.path.splitext(fn)[0]
                    local_prob_sum[image_id] = float(local_prob_sum.get(image_id, 0.0) + float(pr))
                    local_prob_cnt[image_id] = int(local_prob_cnt.get(image_id, 0) + 1)

    # 平均確率（アンサンブル+TTA平均）を計算
    # まず probability を保存しておくと、閾値を変えるだけで再提出できて便利。
    local_pairs_prob = []  # (image_id:int, prob:float)
    for image_id in sorted(local_prob_sum.keys(), key=lambda x: int(x)):
        p = local_prob_sum[image_id] / max(1, local_prob_cnt[image_id])
        local_pairs_prob.append((int(image_id), float(p)))

    # 0/1は最後に閾値で作る
    local_pairs = [(iid, int(prob >= threshold)) for (iid, prob) in local_pairs_prob]

    dist.barrier()
    gathered = [None for _ in range(world_size)] if is_main_process(rank) else None
    gathered_prob = [None for _ in range(world_size)] if is_main_process(rank) else None

    dist.gather_object(local_pairs, gathered, dst=0)
    dist.gather_object(local_pairs_prob, gathered_prob, dst=0)

    if is_main_process(rank):
        all_pairs = []
        for part in gathered:
            if part:
                all_pairs.extend(part)

        all_probs = []
        for part in gathered_prob:
            if part:
                all_probs.extend(part)

        # sort by numeric image_id
        all_pairs.sort(key=lambda x: int(x[0]))
        all_probs.sort(key=lambda x: int(x[0]))

        prediction_dir = os.path.join(output_dir, "Prediction")
        os.makedirs(prediction_dir, exist_ok=True)

        # 1) probabilities (for threshold sweep)
        prob_file_path = os.path.join(prediction_dir, f"sample_submit_probs_t{threshold:.2f}.csv")
        df_prob = pd.DataFrame(all_probs, columns=["image_id", "prob"])
        df_prob.to_csv(prob_file_path, index=False)

        # 2) labels (actual submission)
        submit_file_path = os.path.join(prediction_dir, f"sample_submit_t{threshold:.2f}.csv")
        df = pd.DataFrame(all_pairs, columns=["image_id", "target"])
        df.to_csv(submit_file_path, index=False)

        print(f"prob file: {prob_file_path}")
        print(f"submit file: {submit_file_path} (threshold={threshold})")

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    # torchrun で起動された時だけ torchrun パスに入る。
    # （Windows で env 変数が残留していると、kubernetes.docker.internal に繋ぎに行って警告が出るため）
    is_torchrun = (
        "TORCHELASTIC_RUN_ID" in os.environ
        or ("LOCAL_WORLD_SIZE" in os.environ and "RANK" in os.environ and "WORLD_SIZE" in os.environ)
    )

    if is_torchrun:
        main()
    else:
        # python 直実行なら 2 プロセス spawn して 2GPU/2プロセス学習
        # 念のため torchrun 関連 env を消してから spawn する
        for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "LOCAL_WORLD_SIZE", "TORCHELASTIC_RUN_ID"]:
            os.environ.pop(k, None)

        ws = 2
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            if n_gpu < 2:
                raise RuntimeError(f"2GPU想定ですが GPUが{n_gpu}個しかありません")

        port = find_free_port()
        mp.spawn(main, args=(ws, "127.0.0.1", port), nprocs=ws, join=True)