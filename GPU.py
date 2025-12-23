import os
import platform
import math
import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import timm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=r".*autocast.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*NCCL support.*")


# =========================================================
# GPU チェック
# =========================================================
if not torch.cuda.is_available():
    raise RuntimeError("CUDA が有効ではありません。GPU 環境で実行してください。")

gpu_count = torch.cuda.device_count()
if gpu_count < 2:
    raise RuntimeError("このスクリプトは GPU 2 枚以上が必要です。")


# =========================================================
# 乱数固定
# =========================================================
def set_seed(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# =========================================================
# DDP 初期化 & 後処理
# =========================================================
def setup_distributed():
    """
    torchrun で渡される環境変数から DDP を初期化。
    環境変数が無いときは DDP を使わず、後で DataParallel で 2GPU を使う。
    """
    if "RANK" not in os.environ:
        device = torch.device("cuda:0")
        is_distributed = False
        rank = 0
        world_size = torch.cuda.device_count()
        return is_distributed, rank, world_size, device

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    backend = "nccl"
    if platform.system() == "Windows":
        backend = "gloo"

    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return True, rank, world_size, device


def cleanup_distributed(is_distributed: bool):
    if is_distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# =========================================================
# Utility: unwrap model (DDP/DP対応)
# =========================================================
def unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


# =========================================================
# TensorBoard: matplotlib Figure -> Tensor
# =========================================================
def fig_to_tensor(fig):
    """matplotlib Figure -> torch Tensor(C,H,W) for TB add_image"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    tensor = to_tensor(img)  # (C,H,W) float[0,1]
    plt.close(fig)
    return tensor


def plot_roc_curve_fig(y_true: np.ndarray, y_prob: np.ndarray, title="ROC"):
    """ROC曲線のFigureを返す（片クラスの場合はNone）"""
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (AUC={auc:.4f})")
    plt.grid(True)
    return fig


def plot_confusion_matrix_fig(cm: np.ndarray, class_names=("0", "1"), title="Confusion Matrix"):
    """混同行列Figure"""
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(int(cm[i, j])),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    return fig


# =========================================================
# NaN/Inf サニタイズ（追加）
# =========================================================
def sanitize_binary_arrays(y_true: np.ndarray, y_prob: np.ndarray):
    """
    y_true, y_prob から NaN/Inf を除外して返す。
    y_prob は [0,1] にクリップ。
    """
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).astype(np.float32)

    mask = np.isfinite(y_prob) & np.isfinite(y_true)
    y_true2 = y_true[mask]
    y_prob2 = y_prob[mask]

    y_prob2 = np.clip(y_prob2, 0.0, 1.0).astype(np.float32)
    return y_true2, y_prob2, mask


# =========================================================
# Dataset クラス
# =========================================================
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.data = []
        self._prepare_data()

    def _prepare_data(self):
        for class_label in self.classes:
            class_path = os.path.join(self.root_dir, class_label)
            label_index = self.classes.index(class_label)

            if not os.path.isdir(class_path):
                continue

            for img_file in tqdm(os.listdir(class_path), desc=f"Loading {class_label}"):
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

        if self.transform:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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
        if self.transform:
            image = self.transform(image)
        return image, fn


# =========================================================
# Utility: threshold selection (Accuracy maximize)  ※修正
# =========================================================
def find_best_threshold_by_accuracy(y_true: np.ndarray, y_prob: np.ndarray):
    """
    val上で Accuracy が最大となる閾値を選ぶ。
    - NaN/Inf は除外
    - roc_curve が失敗した場合は linspace の閾値で探索
    """
    y_true, y_prob, _ = sanitize_binary_arrays(y_true, y_prob)

    if y_true.size == 0:
        return 0.5, float("nan")

    # 片クラスしか無いと roc_curve が使えないので 0.5 固定
    if len(np.unique(y_true)) < 2:
        y_pred = (y_prob >= 0.5).astype(np.int32)
        return 0.5, float(accuracy_score(y_true, y_pred))

    # roc_curve の閾値候補（安全化）
    try:
        _, _, thr = roc_curve(y_true, y_prob)
        thr = thr[np.isfinite(thr)]
        thr = np.clip(thr, 0.0, 1.0)
        thr = np.unique(thr)
        if thr.size == 0:
            raise ValueError("Empty thresholds from roc_curve")
    except Exception:
        # fallback：0〜1の等間隔で探索
        thr = np.linspace(0.0, 1.0, 1001, dtype=np.float32)

    best_t = 0.5
    best_acc = -1.0

    for t in thr:
        t = float(t)
        y_pred = (y_prob >= t).astype(np.int32)
        acc = float(accuracy_score(y_true, y_pred))
        if acc > best_acc:
            best_acc = acc
            best_t = t

    return float(best_t), float(best_acc)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    """
    指標一括計算（AUC, Acc, BalancedAcc, Sens, Spec, CM）
    - NaN/Inf を除外してから計算
    """
    y_true, y_prob, _ = sanitize_binary_arrays(y_true, y_prob)

    if y_true.size == 0:
        return {
            "acc": float("nan"),
            "auc": float("nan"),
            "balanced_acc": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        }

    y_pred = (y_prob >= float(threshold)).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape != (2, 2):
        return {
            "acc": float("nan"),
            "auc": float("nan"),
            "balanced_acc": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        }

    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    sensitivity = float(tp / (tp + fn)) if (tp + fn) != 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) != 0 else 0.0
    balanced_acc = 0.5 * (sensitivity + specificity)

    return {
        "acc": acc,
        "auc": auc,
        "balanced_acc": float(balanced_acc),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# =========================================================
# ④ MixUp / CutMix（今回は “画像の線形MixUp” をMixUp/CutMix枠として採用）
# =========================================================
def mixup_or_cutmix_linear(x, y, alpha=1.0, p=0.5):
    """
    p の確率で、lam ~ Beta(alpha, alpha) の線形混合を実施
    戻り:
      mixed_x, (y_a, y_b), lam  or  (x, None, None)
    """
    if np.random.rand() > p:
        return x, None, None

    lam = float(np.random.beta(alpha, alpha))
    b = x.size(0)
    index = torch.randperm(b, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a = y
    y_b = y[index]
    return mixed_x, (y_a, y_b), lam


# =========================================================
# ③ Warmup + Cosine Scheduler（epoch単位）
# =========================================================
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # warmup: 1..warmup_epochs で線形増加
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]

        # cosine: warmup後にcosineで min_lr へ
        denom = max(1, (self.max_epochs - self.warmup_epochs))
        progress = (self.last_epoch - self.warmup_epochs) / denom  # 0..1
        progress = float(np.clip(progress, 0.0, 1.0))

        return [
            self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


# =========================================================
# ② 層別LR（backbone/head）
# =========================================================
def build_optimizer_layerwise_lr(model: nn.Module, wd=1e-4, lr_backbone=1e-4, lr_head=5e-4):
    m = unwrap_model(model)

    backbone_params = []
    head_params = []

    # timm EfficientNetは classifier が多いが、保険で head/fc も拾う
    head_keys = ("classifier", "head", "fc")

    for name, p in m.named_parameters():
        if any(k in name for k in head_keys):
            head_params.append(p)
        else:
            backbone_params.append(p)

    if len(head_params) == 0:
        head_params = []
        backbone_params = list(m.parameters())

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=wd,
    )
    return optimizer


# =========================================================
# 転移学習：freeze/unfreeze
# =========================================================
def set_backbone_trainable(model: nn.Module, trainable: bool):
    m = unwrap_model(model)

    head_keys = ("classifier", "head", "fc")
    for name, p in m.named_parameters():
        if any(k in name for k in head_keys):
            p.requires_grad = True
        else:
            p.requires_grad = bool(trainable)


# =========================================================
# pos_weight 計算（Subset対応）
# =========================================================
def compute_pos_weight_from_subset(dataset: BreastCancerDataset, indices: np.ndarray):
    labels = np.array([dataset.data[i][1] for i in indices], dtype=np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0:
        raise RuntimeError("train に陽性(1)が 0 件です。pos_weight を計算できません。")
    return float(n_neg / n_pos), labels


# =========================================================
# メイン
# =========================================================
def main():
    set_seed(1234)

    # 1) DDP 初期化
    is_distributed, rank, world_size, device = setup_distributed()
    dist_mod = torch.distributed if is_distributed else None

    # 2) データパス
    train_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer_mk2\train"
    test_dir  = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer_mk2\test"
    classes = ["0", "1"]

    IMG_SIZE = 416

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(
            IMG_SIZE,
            scale=(0.9, 1.0),
            translate=(0.1, 0.1),
            ratio=(1, 1.1),
           interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.08), ratio=(0.3, 3.3), value="random"),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 3) Dataset（train全体を読み、KFoldで分割）
    full_train_dataset = BreastCancerDataset(train_dir, classes, train_transforms)
    full_train_dataset_valview = BreastCancerDataset(train_dir, classes, val_test_transforms)

    # 4) StratifiedKFold
    N_FOLDS = 2
    all_labels = np.array([lab for _, lab in full_train_dataset.data], dtype=np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=1234)

    # 5) 学習設定
    global_batch = 32
    accum_steps = 2
    num_epochs = 20
    freeze_epochs = 4

    # warmup + cosine
    warmup_epochs = 3
    min_lr = 1e-6

    # MixUp/CutMix
    mix_alpha = 1.0
    mix_p = 0.5

    # AMP
    use_amp = True
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # TensorBoard（rank0だけ）
    base_log_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_kazma\runs_effb4_acc_kfold"
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_log_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None

    if rank == 0:
        writer.add_text("config", f"""
model=tf_efficientnet_b4_ns
is_distributed={is_distributed}
world_size={world_size}
global_batch={global_batch}
accum_steps={accum_steps}
img_size={IMG_SIZE}
folds={N_FOLDS}
epochs={num_epochs}
freeze_epochs={freeze_epochs}
amp={use_amp}
mixup_cutmix_linear_p={mix_p}
mix_alpha={mix_alpha}
layerwise_lr=backbone:1e-4, head:5e-4
scheduler=WarmupCosine(epoch_warmup={warmup_epochs}, min_lr={min_lr})
train_dir={train_dir}
test_dir={test_dir}
log_dir={log_dir}
""")
        writer.flush()

    fold_states = []
    fold_thresholds = []
    fold_best_accs = []

    # =====================================================
    # Fold Loop
    # =====================================================
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels), start=1):
        if rank == 0:
            print(f"\n========== Fold {fold}/{N_FOLDS} ==========", flush=True)

        train_subset = torch.utils.data.Subset(full_train_dataset, train_idx.tolist())
        val_subset   = torch.utils.data.Subset(full_train_dataset_valview, val_idx.tolist())

        # pos_weight（fold train subsetから計算）
        if rank == 0:
            pos_weight_value, labels_fold_train = compute_pos_weight_from_subset(full_train_dataset, train_idx)
            labels_fold_val = np.array([full_train_dataset.data[i][1] for i in val_idx], dtype=np.int64)
            print("=== Class count (fold) ===", flush=True)
            print("train:", np.bincount(labels_fold_train, minlength=2), flush=True)
            print("val  :", np.bincount(labels_fold_val,   minlength=2), flush=True)
        else:
            
            pos_weight_value = 1.0

        if is_distributed:
            pos_w_tensor = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
            dist_mod.broadcast(pos_w_tensor, src=0)
            pos_weight_value = float(pos_w_tensor.item())

        # Sampler / batch
        if is_distributed:
            train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler   = DistributedSampler(val_subset,   num_replicas=world_size, rank=rank, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        if is_distributed:
            batch_per_gpu = max(1, global_batch // world_size)
        else:
            batch_per_gpu = global_batch

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_per_gpu,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_per_gpu,
            shuffle=False,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )

        # rank0が“全件val”を評価する用
        eval_val_loader = None
        if rank == 0:
            eval_val_loader = DataLoader(
                val_subset,
                batch_size=128,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

        # モデル
        model = timm.create_model(
            "tf_efficientnet_b4_ns",
            pretrained=True,
            num_classes=1,
        ).to(device)

        if is_distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device.index] if device.type == "cuda" else None
            )
            if rank == 0:
                print(f"[Rank {rank}] use DDP on device {device}", flush=True)
        else:
            gpu_ids = list(range(min(2, torch.cuda.device_count())))
            model = nn.DataParallel(model, device_ids=gpu_ids)
            if rank == 0:
                print(f"Use DataParallel on GPUs: {gpu_ids}", flush=True)

        # Loss / Optimizer / Scheduler
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))

        optimizer = build_optimizer_layerwise_lr(
            model, wd=1e-4, lr_backbone=1e-4, lr_head=5e-4
        )

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=num_epochs,
            min_lr=min_lr
        )

        best_val_acc = -1e9
        best_state_dict = None
        best_threshold = 0.5
        best_metrics_snapshot = None

        epoch_iter = tqdm(range(num_epochs), desc=f"Fold {fold} Training") if rank == 0 else range(num_epochs)

        for epoch in epoch_iter:
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if epoch < freeze_epochs:
                set_backbone_trainable(model, False)
            else:
                set_backbone_trainable(model, True)

            # ----- Train -----
            model.train()
            running_loss = 0.0
            correct_05, total = 0.0, 0.0

            optimizer.zero_grad(set_to_none=True)
            last_step_idx = -1

            for step_idx, (inputs, labels) in enumerate(train_loader):
                last_step_idx = step_idx
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.float().to(device).view(-1, 1)

                # MixUp/CutMix
                mixed_inputs, labels_pair, lam = mixup_or_cutmix_linear(inputs, labels, alpha=mix_alpha, p=mix_p)
                mixed = labels_pair is not None
                if mixed:
                    inputs_for_model = mixed_inputs
                    y_a, y_b = labels_pair
                else:
                    inputs_for_model = inputs

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(inputs_for_model).view(-1, 1)
                    if mixed:
                        loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
                    else:
                        loss = criterion(outputs, labels)

                    loss = loss / accum_steps

                # 追加：loss NaN/Inf 検知（発散対策）
                if not torch.isfinite(loss):
                    if rank == 0:
                        print(f"[WARN] Non-finite loss detected (Fold{fold}, epoch={epoch}, step={step_idx}). Skipping.", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()

                if (step_idx + 1) % accum_steps == 0:
                    # 追加：勾配クリップ（AMP時は unscale 必須）
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                bs = inputs.size(0)
                running_loss += loss.item() * bs * accum_steps

                # train acc@0.5
                with torch.no_grad():
                    preds = (torch.sigmoid(outputs) >= 0.5).float()
                    ref_labels = y_a if mixed else labels
                    correct_05 += (preds == ref_labels).sum().item()
                    total += bs

            # 勾配累積の端数を捨てない
            if last_step_idx >= 0 and (last_step_idx + 1) % accum_steps != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # DDP集計
            if is_distributed:
                loss_tensor = torch.tensor(running_loss, device=device)
                correct_tensor = torch.tensor(correct_05, device=device)
                total_tensor = torch.tensor(total, device=device)

                dist_mod.all_reduce(loss_tensor, op=dist_mod.ReduceOp.SUM)
                dist_mod.all_reduce(correct_tensor, op=dist_mod.ReduceOp.SUM)
                dist_mod.all_reduce(total_tensor, op=dist_mod.ReduceOp.SUM)

                running_loss = loss_tensor.item()
                correct_05 = correct_tensor.item()
                total = total_tensor.item()

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc_05 = correct_05 / max(1.0, total)

            # ----- Validation（loss/acc@0.5を分散集計） -----
            model.eval()
            val_loss = 0.0
            val_correct_05, val_total = 0.0, 0.0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.float().to(device).view(-1, 1)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = model(inputs).view(-1, 1)
                        loss = criterion(outputs, labels)

                    bs = inputs.size(0)
                    val_loss += loss.item() * bs

                    preds = (torch.sigmoid(outputs) >= 0.5).float()
                    val_correct_05 += (preds == labels).sum().item()
                    val_total += bs

            if is_distributed:
                loss_tensor = torch.tensor(val_loss, device=device)
                correct_tensor = torch.tensor(val_correct_05, device=device)
                total_tensor = torch.tensor(val_total, device=device)

                dist_mod.all_reduce(loss_tensor, op=dist_mod.ReduceOp.SUM)
                dist_mod.all_reduce(correct_tensor, op=dist_mod.ReduceOp.SUM)
                dist_mod.all_reduce(total_tensor, op=dist_mod.ReduceOp.SUM)

                val_loss = loss_tensor.item()
                val_correct_05 = correct_tensor.item()
                val_total = total_tensor.item()

            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_epoch_acc_05 = val_correct_05 / max(1.0, val_total)

            # scheduler（epoch end）
            scheduler.step()

            # rank0だけ：全件valで確率集計 → best threshold（Acc最大） + ROC/CM 可視化
            if rank == 0:
                model.eval()
                all_labels_fold, all_probs_fold = [], []
                nan_batches = 0
                total_batches = 0

                with torch.no_grad():
                    for inputs, labels in eval_val_loader:
                        total_batches += 1
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.float().to(device).view(-1, 1)

                        with torch.amp.autocast("cuda", enabled=use_amp):
                            outputs = model(inputs).view(-1, 1)

                        # 重要：logits を先に潰す（sigmoidの前）
                        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=50.0, neginf=-50.0)
                        probs = torch.sigmoid(outputs)

                        if torch.isnan(probs).any() or torch.isinf(probs).any():
                            nan_batches += 1
                            probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)

                        all_probs_fold.extend(probs.detach().cpu().numpy().flatten())
                        all_labels_fold.extend(labels.detach().cpu().numpy().flatten())

                y_true = np.array(all_labels_fold, dtype=np.int32)
                y_prob = np.array(all_probs_fold, dtype=np.float32)

                # 念のため最終sanitize（roc_curve落ち防止）
                y_true_s, y_prob_s, _ = sanitize_binary_arrays(y_true, y_prob)

                if y_true_s.size == 0:
                    epoch_best_t = 0.5
                    epoch_metrics = {
                        "acc": float("nan"),
                        "auc": float("nan"),
                        "balanced_acc": float("nan"),
                        "sensitivity": float("nan"),
                        "specificity": float("nan"),
                        "tn": 0, "fp": 0, "fn": 0, "tp": 0,
                    }
                else:
                    epoch_best_t, _ = find_best_threshold_by_accuracy(y_true_s, y_prob_s)
                    epoch_metrics = compute_binary_metrics(y_true_s, y_prob_s, threshold=epoch_best_t)

                current_bestt_acc = float(epoch_metrics["acc"]) if not np.isnan(epoch_metrics["acc"]) else -1.0

                if nan_batches > 0:
                    print(f"[WARN] NaN/Inf probs detected in {nan_batches}/{total_batches} eval batches (Fold{fold}, epoch={epoch})", flush=True)

                # TB: Scalars
                if writer is not None:
                    writer.add_scalars(f"Fold{fold}/Loss", {"train": epoch_loss, "val": val_epoch_loss}, epoch)
                    writer.add_scalars(f"Fold{fold}/Accuracy@0.5", {"train": epoch_acc_05, "val": val_epoch_acc_05}, epoch)
                    writer.add_scalar(f"Fold{fold}/VAL_BestThreshold(Acc)", float(epoch_best_t), epoch)
                    writer.add_scalar(f"Fold{fold}/VAL_Acc@best_t", float(current_bestt_acc), epoch)
                    writer.add_scalar(
                        f"Fold{fold}/VAL_AUC(ref)",
                        float(epoch_metrics["auc"]) if not np.isnan(epoch_metrics["auc"]) else 0.0,
                        epoch
                    )
                    writer.add_scalar(f"Fold{fold}/LR_group0(backbone)", optimizer.param_groups[0]["lr"], epoch)
                    writer.add_scalar(f"Fold{fold}/LR_group1(head)", optimizer.param_groups[1]["lr"], epoch)

                    # ROC曲線（sanitize後の配列で）
                    roc_fig = plot_roc_curve_fig(y_true_s, y_prob_s, title=f"Fold{fold} ROC (epoch={epoch})")
                    if roc_fig is not None:
                        writer.add_image(f"Fold{fold}/ROC_curve", fig_to_tensor(roc_fig), global_step=epoch)
                    else:
                        writer.add_text(f"Fold{fold}/ROC_curve", "Only one class in y_true; ROC unavailable.", epoch)

                    # 混同行列（best threshold）
                    y_pred_best = (y_prob_s >= float(epoch_best_t)).astype(np.int32)
                    cm = confusion_matrix(y_true_s, y_pred_best)
                    cm_fig = plot_confusion_matrix_fig(
                        cm,
                        class_names=("0", "1"),
                        title=f"Fold{fold} CM (t={epoch_best_t:.3f}) epoch={epoch}"
                    )
                    writer.add_image(f"Fold{fold}/Confusion_matrix", fig_to_tensor(cm_fig), global_step=epoch)

                    writer.add_scalars(f"Fold{fold}/VAL_Accuracy", {
                        "acc@0.5": float(val_epoch_acc_05),
                        "acc@best_t": float(current_bestt_acc),
                    }, epoch)

                    writer.flush()

                if hasattr(epoch_iter, "set_postfix"):
                    epoch_iter.set_postfix(
                        loss=float(epoch_loss),
                        train_acc_05=float(epoch_acc_05),
                        val_loss=float(val_epoch_loss),
                        val_acc_05=float(val_epoch_acc_05),
                        val_acc_bestt=float(current_bestt_acc),
                        t=float(epoch_best_t),
                    )

                # best更新（fold内）
                if best_metrics_snapshot is None or current_bestt_acc > float(best_metrics_snapshot["acc"]):
                    best_threshold = float(epoch_best_t)
                    best_metrics_snapshot = dict(epoch_metrics)

                if current_bestt_acc > best_val_acc:
                    best_val_acc = current_bestt_acc
                    sd = unwrap_model(model).state_dict()
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in sd.items()}

        # fold終わり：rank0だけ保存用に保持
        if rank == 0:
            print(f"[Fold {fold}] best_val_acc@best_t = {best_val_acc:.6f}, best_threshold={best_threshold:.6f}", flush=True)
            fold_best_accs.append(float(best_val_acc))
            fold_thresholds.append(float(best_threshold))
            fold_states.append(best_state_dict)

        if is_distributed:
            dist_mod.barrier()

    # =====================================================
    # rank0：fold平均で test 推論 → 提出CSV作成
    # =====================================================
    if rank == 0:
        output_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_kazma"
        os.makedirs(output_dir, exist_ok=True)
        weight_dir = os.path.join(output_dir, "Weight")
        os.makedirs(weight_dir, exist_ok=True)

        for i, sd in enumerate(fold_states, start=1):
            if sd is None:
                continue
            torch.save(sd, os.path.join(weight_dir, f"fold{i}_best.pth"))

        best_threshold_mean = float(np.mean(fold_thresholds)) if len(fold_thresholds) > 0 else 0.5

        print("\n=== CV Summary ===")
        print("fold accs:", fold_best_accs)
        print("mean acc :", float(np.mean(fold_best_accs)) if len(fold_best_accs) else float("nan"))
        print("fold thresholds:", fold_thresholds)
        print("mean threshold :", best_threshold_mean)

        if writer is not None:
            writer.add_text("cv_summary", f"""
fold_accs={fold_best_accs}
mean_acc={float(np.mean(fold_best_accs)) if len(fold_best_accs) else float('nan')}
fold_thresholds={fold_thresholds}
mean_threshold={best_threshold_mean}
""")
            writer.flush()

        test_dataset = TestDataset(test_dir, val_test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

        all_test_probs = None
        all_filenames = None

        for fi, sd in enumerate(fold_states, start=1):
            if sd is None:
                continue

            infer_device = torch.device("cuda:0")
            infer_model = timm.create_model("tf_efficientnet_b4_ns", pretrained=False, num_classes=1).to(infer_device)
            infer_model.load_state_dict(sd, strict=True)
            infer_model.eval()

            probs_this_fold = []
            filenames_this = []

            with torch.no_grad():
                for images, filenames in tqdm(test_loader, desc=f"Test Prediction (fold{fi})"):
                    images = images.to(infer_device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = infer_model(images).view(-1)

                    # 推論側も安全に
                    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=50.0, neginf=-50.0)
                    probs = torch.sigmoid(outputs).detach().cpu().numpy().astype(np.float32)
                    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)

                    probs_this_fold.append(probs)
                    filenames_this.extend(list(filenames))

            probs_this_fold = np.concatenate(probs_this_fold, axis=0)

            if all_test_probs is None:
                all_test_probs = probs_this_fold
                all_filenames = filenames_this
            else:
                all_test_probs += probs_this_fold

            del infer_model
            torch.cuda.empty_cache()

        if all_test_probs is None:
            raise RuntimeError("fold_states が空です。学習が正常に完了していない可能性があります。")

        all_test_probs = all_test_probs / float(len(fold_states))

        preds = (all_test_probs >= best_threshold_mean).astype(int)

        predictions = []
        for fn, p in zip(all_filenames, preds):
            image_id = os.path.splitext(fn)[0]
            predictions.append((image_id, int(p)))

        prediction_dir = os.path.join(output_dir, "Prediction")
        os.makedirs(prediction_dir, exist_ok=True)

        submit_file_path = os.path.join(prediction_dir, "sample_submit_kazma_1222_GPU2_1.csv")
        df = pd.DataFrame(predictions, columns=["image_id", "target"])
        df.to_csv(submit_file_path, index=False)

        print(f"\nサブミットファイルが {submit_file_path} に作成されました。")
        print(f"TensorBoard logdir: {log_dir}")

        if writer is not None:
            writer.close()

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()