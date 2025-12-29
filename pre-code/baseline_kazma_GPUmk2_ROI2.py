import os
import sys
from pathlib import Path

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

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from torch.utils.tensorboard import SummaryWriter
import timm
import cv2


# ===========================
# Youden's J で閾値決定
# ===========================
def threshold_by_youden_j(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Youden's J = TPR - FPR を最大化する閾値を返す
    """
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_prob = np.asarray(y_prob).astype(np.float64).ravel()

    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    return float(thr[idx])


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
# DDP 初期化（Windows対応版）
# ===========================
def ddp_setup():
    if "RANK" not in os.environ:
        raise RuntimeError("❌ このスクリプトは torchrun --nproc_per_node=2 専用です")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if world_size != 2:
        raise RuntimeError(f"❌ WORLD_SIZE={world_size} です。必ず2プロセスで起動してください")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    backend = "gloo" if os.name == "nt" else "nccl"
    dist.init_process_group(backend=backend, init_method="env://")
    return rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def cleanup():
    dist.destroy_process_group()


# ===========================
# ★追加：pos_weight = Nneg/Npos を計算
# ===========================
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}


def count_class_images(root_dir: str, neg_class: str = "0", pos_class: str = "1"):
    """
    root_dir/neg_class, root_dir/pos_class 配下の画像枚数を数える（再帰）
    """
    root = Path(root_dir)

    def count_in_class(cname: str) -> int:
        d = root / cname
        if not d.exists():
            return 0
        n = 0
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                n += 1
        return n

    return count_in_class(neg_class), count_in_class(pos_class)


def compute_pos_weight(n_neg: int, n_pos: int) -> float:
    if n_pos == 0:
        raise ValueError("pos(=1) の枚数が 0 です。pos_weight を計算できません。")
    return float(n_neg) / float(n_pos)


# ===========================
# YOLOX: ROI cropper
# （まずは動く簡易版：416へ単純リサイズして推論→元画像へスケールで戻す）
# ===========================
def make_yolox_roi_cropper(
    yolox_model,
    postprocess_fn,
    test_size=(416, 416),
    conf=0.3,
    nms=0.65,
    pad_ratio=0.05,
):
    device = next(yolox_model.parameters()).device

    def cropper(pil_img: Image.Image) -> Image.Image:
        img_rgb = np.array(pil_img)  # HWC, RGB
        H, W = img_rgb.shape[:2]

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
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

        # 416座標 -> 元画像座標へ戻す（簡易：単純リサイズのみ想定）
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
# Dataset クラス（YOLOX ROI対応）
# ===========================
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None, roi_detector=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.roi_detector = roi_detector
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
                    (".png", ".jpg", ".jpeg", ".bmp", ".gif")
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

        if self.roi_detector is not None:
            try:
                image = self.roi_detector(image)
            except Exception:
                pass

        if self.transform:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None, roi_detector=None):
        self.root_dir = root_dir
        self.transform = transform
        self.roi_detector = roi_detector
        self.image_files = [
            f
            for f in sorted(os.listdir(self.root_dir))
            if os.path.isfile(os.path.join(self.root_dir, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fn = self.image_files[idx]
        path = os.path.join(self.root_dir, fn)
        image = Image.open(path).convert("RGB")

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
        padded[: t.numel()] = t

    gathered = [torch.zeros((max_n,), device=device, dtype=t.dtype) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    out = []
    for g, n in zip(gathered, sizes):
        if n > 0:
            out.append(g[:n])

    if len(out) == 0:
        return torch.empty((0,), device=device, dtype=t.dtype)
    return torch.cat(out, dim=0)


def main():
    rank, local_rank, world_size = ddp_setup()
    set_seed(1234, rank=rank)

    device = torch.device(f"cuda:{local_rank}")

    train_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer_300GB\train"
    val_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\valid"
    test_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\test"

    log_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_kazma\log_GPU2"
    writer = SummaryWriter(log_dir=log_dir) if is_main_process(rank) else None

    classes = ["0", "1"]

    # ===========================
    # YOLOX setup
    # ===========================
    yolox_root = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\YOLOX"
    if yolox_root not in sys.path:
        sys.path.insert(0, yolox_root)

    from yolox.exp import get_exp
    from yolox.utils import postprocess

    exp_file = os.path.join(yolox_root, r"exps\custom\yolox_nano_breast_roi_416.py")
    ckpt_file = os.path.join(yolox_root, r"YOLOX_outputs\yolox_base\latest_ckpt.pth")

    yexp = get_exp(exp_file, None)
    yexp.test_conf = 0.3
    yexp.nmsthre = 0.65
    yexp.test_size = (416, 416)

    yolox_model = yexp.get_model().to(device)
    yolox_model.eval()

    ckpt = torch.load(ckpt_file, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    yolox_model.load_state_dict(state, strict=False)

    roi_cropper = make_yolox_roi_cropper(
        yolox_model=yolox_model,
        postprocess_fn=postprocess,
        test_size=(416, 416),
        conf=0.3,
        nms=0.65,
        pad_ratio=0.05,
    )

    # ===========================
    # Transform
    # ===========================
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=384, scale=(512 / 640, 1.0), ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomInvert(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_test_transforms = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.Resize(384),
            transforms.RandomInvert(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = BreastCancerDataset(train_dir, classes, train_transforms, roi_detector=roi_cropper)
    val_dataset = BreastCancerDataset(val_dir, classes, val_test_transforms, roi_detector=roi_cropper)

    # ===========================
    # ★追加：pos_weight を train の枚数から計算（rank0で数えてbroadcast）
    # ===========================
    pos_weight_value = None
    if is_main_process(rank):
        train_neg, train_pos = count_class_images(train_dir, neg_class="0", pos_class="1")
        val_neg, val_pos = count_class_images(val_dir, neg_class="0", pos_class="1")

        pos_weight_value = compute_pos_weight(train_neg, train_pos)

        print(f"[Count] train: neg={train_neg}, pos={train_pos}")
        print(f"[Count] valid: neg={val_neg}, pos={val_pos}")
        print(f"[pos_weight] Nneg/Npos = {pos_weight_value}")

    # broadcast（全rankで同じ値にする）
    pos_w_t = torch.tensor([0.0], device=device, dtype=torch.float32)
    if is_main_process(rank):
        pos_w_t.fill_(float(pos_weight_value))
    dist.broadcast(pos_w_t, src=0)
    pos_weight_value = float(pos_w_t.item())

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, sampler=val_sampler, num_workers=0, pin_memory=True)

    model = timm.create_model("tf_efficientnet_b4_ns", pretrained=True, num_classes=1).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    pos_weight = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    num_epochs = 10

    # ===========================
    # best (AUC最大 epoch) の管理
    # ===========================
    best_auc = -1.0
    best_epoch = -1
    best_thr = 0.5
    best_state_dict = None  # rank0のみ保持

    if writer is not None:
        writer.add_text(
            "config",
            f"""
model=tf_efficientnet_b4_ns
batch_size=8
lr=1e-4
weight_decay=1e-4
epochs={num_epochs}
pos_weight(train Nneg/Npos)={pos_weight_value}
threshold=VAL_YoudenJ (per-epoch), best selected by max AUC
log_dir={log_dir}
world_size={world_size}
roi=yolox_nano_breast_roi_416 (416)
""",
        )

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        # -------------------------------
        # Train
        # -------------------------------
        model.train()
        running_loss = 0.0
        correct, total = 0, 0  # train_accは参考値として0.5固定で算出

        global_step = epoch * len(train_loader)
        epoch_bar = (
            tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Train", leave=False)
            if is_main_process(rank)
            else train_loader
        )

        for inputs, labels in epoch_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs).view(-1, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            if writer is not None:
                writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            global_step += 1

            preds = (torch.sigmoid(outputs) >= 0.5).float()
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
        # Validation（閾値はYouden、bestはAUC最大）
        # -------------------------------
        model.eval()
        val_loss = 0.0
        local_probs = []
        local_labels = []

        with torch.no_grad():
            val_bar = (
                tqdm(val_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Val", leave=False)
                if is_main_process(rank)
                else val_loader
            )

            for inputs, labels in val_bar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True).view(-1, 1)

                outputs = model(inputs).view(-1, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(outputs)
                local_probs.append(probs.view(-1))
                local_labels.append(labels.view(-1))

        # val loss 集計
        val_loss_t = torch.tensor([val_loss], device=device)
        dist.all_reduce(val_loss_t, op=dist.ReduceOp.SUM)
        val_epoch_loss = val_loss_t.item() / max(1, len(val_dataset))

        # probs/labels gather
        local_probs_t = torch.cat(local_probs, dim=0) if len(local_probs) else torch.empty((0,), device=device)
        local_labels_t = torch.cat(local_labels, dim=0) if len(local_labels) else torch.empty((0,), device=device)

        gathered_probs = all_gather_1d_float_tensor(local_probs_t)
        gathered_labels = all_gather_1d_float_tensor(local_labels_t)

        # rank0で評価→best更新、best_thrをbroadcast
        thr_t = torch.zeros((), device=device, dtype=torch.float32)
        best_auc_t = torch.zeros((), device=device, dtype=torch.float32)

        if is_main_process(rank):
            y_prob = gathered_probs.detach().cpu().numpy()
            y_true = gathered_labels.detach().cpu().numpy()

            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
            thr = threshold_by_youden_j(y_true, y_prob)

            y_pred = (y_prob >= thr).astype(np.int64)
            acc = accuracy_score(y_true, y_pred)

            cm = confusion_matrix(y_true, y_pred)
            if cm.size == 4:
                TN, FP, FN, TP = cm.ravel()
                sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0.0
                specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0
            else:
                sensitivity, specificity = float("nan"), float("nan")

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"train_loss={epoch_loss:.6f} train_acc(0.5)={epoch_acc:.6f} | "
                f"val_loss={val_epoch_loss:.6f} | "
                f"AUC={auc} thr(Youden)={thr} Sens={sensitivity} Spec={specificity} Acc={acc}"
            )

            if writer is not None:
                writer.add_scalars("Loss", {"train": epoch_loss, "val": val_epoch_loss}, epoch)
                writer.add_scalar("Train/Acc_thr0.5", epoch_acc, epoch)
                writer.add_scalar("Val/AUC", auc if auc == auc else 0.0, epoch)
                writer.add_scalar("Val/Threshold_YoudenJ", thr if thr == thr else 0.0, epoch)
                writer.add_scalar("Val/Sensitivity", sensitivity if sensitivity == sensitivity else 0.0, epoch)
                writer.add_scalar("Val/Specificity", specificity if specificity == specificity else 0.0, epoch)
                writer.add_scalar("Val/Accuracy", acc if acc == acc else 0.0, epoch)

                if cm.size == 4:
                    fig = plt.figure()
                    plt.imshow(cm, interpolation="nearest")
                    plt.title("Confusion Matrix (val, thr=Youden)")
                    plt.colorbar()
                    tick_marks = np.arange(2)
                    plt.xticks(tick_marks, ["0", "1"])
                    plt.yticks(tick_marks, ["0", "1"])
                    plt.xlabel("Pred")
                    plt.ylabel("True")
                    for i in range(2):
                        for j in range(2):
                            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
                    writer.add_figure("Val/ConfusionMatrix_Youden", fig, epoch)
                    plt.close(fig)

            # best更新（AUC最大）
            if (auc == auc) and (auc > best_auc):
                best_auc = float(auc)
                best_epoch = int(epoch)
                best_thr = float(thr)
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.module.state_dict().items()}
                print(f"[BEST UPDATE] epoch={best_epoch+1} best_auc={best_auc} best_thr(Youden)={best_thr}")

            thr_t.fill_(best_thr)
            best_auc_t.fill_(best_auc)

        dist.broadcast(thr_t, src=0)
        dist.broadcast(best_auc_t, src=0)
        best_thr = float(thr_t.item())
        best_auc = float(best_auc_t.item())

        dist.barrier()

    if writer is not None:
        writer.close()

    # ===========================
    # best 重み保存（rank0のみ）
    # ===========================
    output_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_kazma\GPU2"
    weight_dir = os.path.join(output_dir, "Weight")
    best_ckpt_path = os.path.join(weight_dir, "breast_cancer_model_ddp_best.pth")

    dist.barrier()
    if is_main_process(rank):
        os.makedirs(weight_dir, exist_ok=True)

        if best_state_dict is None:
            best_state_dict = {k: v.detach().cpu() for k, v in model.module.state_dict().items()}

        torch.save(best_state_dict, best_ckpt_path)
        print(
            f"Saved BEST weights to: {best_ckpt_path} "
            f"(best_epoch={best_epoch+1}, best_auc={best_auc}, best_thr={best_thr})"
        )

    dist.barrier()

    # ===========================
    # テスト推論（best重み + best_thr）
    # ===========================
    torch.cuda.empty_cache()

    # 全rankでbest重みをロード
    state_cpu = torch.load(best_ckpt_path, map_location="cpu")
    model.module.load_state_dict(state_cpu, strict=True)
    dist.barrier()

    test_dataset = TestDataset(test_dir, val_test_transforms, roi_detector=roi_cropper)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=8, sampler=test_sampler, num_workers=0, pin_memory=True)

    model.eval()
    local_pairs = []

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Test Prediction", leave=False) if is_main_process(rank) else test_loader
        for images, filenames in test_bar:
            images = images.to(device, non_blocking=True)
            outputs = model(images).view(-1)
            probs = torch.sigmoid(outputs)

            preds = (probs >= best_thr).to(torch.int64).detach().cpu().numpy().tolist()

            for fn, p in zip(filenames, preds):
                image_id = os.path.splitext(fn)[0]
                local_pairs.append((image_id, int(p)))

    dist.barrier()
    gathered = [None for _ in range(world_size)] if is_main_process(rank) else None
    dist.gather_object(local_pairs, gathered, dst=0)

    if is_main_process(rank):
        all_pairs = []
        for part in gathered:
            if part:
                all_pairs.extend(part)

        all_pairs.sort(key=lambda x: x[0])

        prediction_dir = os.path.join(output_dir, "Prediction")
        os.makedirs(prediction_dir, exist_ok=True)

        submit_file_path = os.path.join(prediction_dir, "sample_submit_kazma_1224_GPU2_roi2.csv")
        df = pd.DataFrame(all_pairs, columns=["image_id", "target"])
        df.to_csv(submit_file_path, index=False)
        print(f"サブミットファイルが {submit_file_path} に作成されました。 (best_thr={best_thr})")

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    main()