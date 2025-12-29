import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import timm
from torchvision import transforms

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# GPU 1枚使用 / 閾値 0.5


# -------------------------
# 乱数固定
# -------------------------

def set_seed(seed: int = 1234) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# -------------------------
# Dataset
# -------------------------

class BreastCancerDataset(Dataset):
    def __init__(self, root_dir: str, classes: list[str], transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.data: list[tuple[str, int]] = []
        self._prepare_data()

    def _prepare_data(self) -> None:
        for class_label in self.classes:
            class_path = os.path.join(self.root_dir, class_label)
            if not os.path.isdir(class_path):
                continue

            label_index = self.class_to_idx[class_label]
            for img_file in tqdm(sorted(os.listdir(class_path)), desc=f"Loading {class_label}"):
                img_full_path = os.path.join(class_path, img_file)
                if os.path.isfile(img_full_path) and img_file.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".gif")
                ):
                    self.data.append((img_full_path, label_index))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert("L")  # グレースケール
        except Exception:
            image = Image.new("L", (224, 224))

        if self.transform:
            image = self.transform(image)

        # BCEWithLogitsLoss 用に float (0/1) で返す
        return image, torch.tensor(label, dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [
            f
            for f in sorted(os.listdir(self.root_dir))
            if os.path.isfile(os.path.join(self.root_dir, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        fn = self.image_files[idx]
        path = os.path.join(self.root_dir, fn)
        image = Image.open(path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, fn


# -------------------------
# mean/std 計算（グレースケール 1ch）
# ※ ここでは Augmentation を入れない（ブレるため）
# -------------------------

def compute_mean_std_grayscale(
    root_dir: str,
    classes: list[str],
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[float, float]:
    stat_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # [1, H, W]
        ]
    )

    stat_dataset = BreastCancerDataset(root_dir, classes, transform=stat_transform)
    stat_loader = DataLoader(
        stat_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    n_pixels = 0
    channel_sum = 0.0
    channel_sum_sq = 0.0

    for x, _ in tqdm(stat_loader, desc="Computing mean/std"):
        # x: [B, 1, H, W]
        x = x.float()
        b, c, h, w = x.shape
        pixels = b * h * w
        n_pixels += pixels
        channel_sum += x.sum().item()
        channel_sum_sq += (x ** 2).sum().item()

    mean = channel_sum / n_pixels
    var = (channel_sum_sq / n_pixels) - (mean ** 2)
    std = float(var ** 0.5)
    return float(mean), std


def select_threshold_sensitivity_first(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_specificity: float = 0.80,
    step: float = 0.01,
) -> tuple[float, dict]:
    """Choose a probability threshold prioritizing Sensitivity (Recall for positive class).

    Strategy:
      1) Maximize Sensitivity
      2) Among ties, maximize Specificity
      3) Among ties, maximize Accuracy

    `min_specificity` prevents the trivial solution (threshold -> 0) that makes sensitivity 1.0.
    """
    best_t = 0.5
    best = {
        "sensitivity": -1.0,
        "specificity": -1.0,
        "accuracy": -1.0,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }

    # Guard: ensure 1D arrays
    y_true = y_true.astype(np.int64).reshape(-1)
    y_prob = y_prob.astype(np.float64).reshape(-1)

    thresholds = np.arange(0.0, 1.0 + 1e-9, step)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int64)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        if spec < min_specificity:
            continue

        better = (
            (sens > best["sensitivity"]) or
            (sens == best["sensitivity"] and spec > best["specificity"]) or
            (sens == best["sensitivity"] and spec == best["specificity"] and acc > best["accuracy"])
        )

        if better:
            best_t = float(t)
            best = {
                "sensitivity": float(sens),
                "specificity": float(spec),
                "accuracy": float(acc),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }

    return best_t, best


def main() -> None:
    set_seed()

    # -------------------------
    # データパス
    # -------------------------
    train_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer_mk2\train"
    val_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer_mk2\valid"
    test_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer_mk2\test"

    log_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_kazma\runs_resnet50d"
    writer = SummaryWriter(log_dir=log_dir)

    classes = ["0", "1"]
    num_workers = 4
    THRESHOLD = 0.10  # 検出率（Sensitivity）重視で固定する閾値

    # -------------------------
    # mean/std をデータから算出
    # -------------------------
    mean, std = compute_mean_std_grayscale(
        train_dir, classes, batch_size=128, num_workers=num_workers
    )
    print(f"[computed] mean={mean:.6f}, std={std:.6f}")

    # -------------------------
    # transforms
    # -------------------------
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, fill=0),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=0,
                fill=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )

    val_test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )

    # -------------------------
    # Dataset / DataLoader
    # -------------------------
    train_dataset = BreastCancerDataset(train_dir, classes, transform=train_transforms)
    val_dataset = BreastCancerDataset(val_dir, classes, transform=val_test_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # -------------------------
    # モデル
    # -------------------------
    model = timm.create_model(
        "resnet50d",
        pretrained=True,
        in_chans=1,
        num_classes=1,  # logit 1つ
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -------------------------
    # Loss / Optimizer
    # -------------------------
    # pos_weight: クラス不均衡がある時に有効（0が多いなら >1 になる）
    num_pos = sum(label == 1 for _, label in train_dataset.data)
    num_neg = sum(label == 0 for _, label in train_dataset.data)
    pos_weight_value = (num_neg / num_pos) if num_pos > 0 else 1.0

    pos_weight = torch.tensor([pos_weight_value], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # -------------------------
    # Training
    # -------------------------
    num_epochs = 13
    epoch_bar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs).view(-1, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) >= THRESHOLD).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).view(-1, 1)

                outputs = model(inputs).view(-1, 1)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * inputs.size(0)

                preds = (torch.sigmoid(outputs) >= THRESHOLD).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_acc = val_correct / val_total

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)

        epoch_bar.set_postfix(loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)

    writer.close()

    # -------------------------
    # 重み保存
    # -------------------------
    output_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_kazma_1215"
    os.makedirs(output_dir, exist_ok=True)

    weight_dir = os.path.join(output_dir, "Weight")
    os.makedirs(weight_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(weight_dir, "breast_cancer_model.pth"))

    # -------------------------
    # 評価（AUCなど） + 閾値最適化（Sensitivity最優先）
    # -------------------------
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1)

            outputs = model(inputs).view(-1, 1)
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    y_true = np.array(all_labels).astype(int)
    y_prob = np.array(all_probs)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    # --- 閾値を固定して評価（Sensitivity重視） ---
    y_pred = (y_prob >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    print("[threshold] policy=Fixed (Sensitivity-first)")
    print(f"[threshold] used={THRESHOLD:.2f}")
    print("Accuracy:", acc)
    print("AUC:", auc)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)

    # -------------------------
    # テスト推論 → サブミット
    # -------------------------
    test_dataset = TestDataset(test_dir, transform=val_test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    predictions = []
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Test Prediction"):
            images = images.to(device, non_blocking=True)
            outputs = model(images).view(-1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= THRESHOLD).long().cpu().numpy().astype(int)

            for fn, p in zip(filenames, preds):
                image_id = os.path.splitext(fn)[0]
                predictions.append((image_id, p))

    prediction_dir = os.path.join(output_dir, "Prediction")
    os.makedirs(prediction_dir, exist_ok=True)

    submit_file_path = os.path.join(prediction_dir, "sample_submit_kazma_1215.csv")
    df = pd.DataFrame(predictions, columns=["image_id", "target"])
    df.to_csv(submit_file_path, index=False)

    print(f"[submit] used_threshold={THRESHOLD:.2f}")
    print(f"サブミットファイルが {submit_file_path} に作成されました。")


if __name__ == "__main__":
    # Windows multiprocessing safety for DataLoader workers
    import multiprocessing as mp

    mp.freeze_support()
    main()
