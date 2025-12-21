import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import timm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from torchvision.transforms import ColorJitter

# ===========================
# 乱数固定
# ===========================
def set_seed(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

# ===========================
# データパス
# ===========================
train_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\train"
val_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\valid"
test_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\test"

classes = ["0", "1"]

# ===========================
# 画像変換
# ===========================
from torchvision import transforms
import numpy as np # transforms.RandomAffine の 'degrees' に利用

# ===========================
# 画像変換（強化版）
# ===========================
PATCH = 224

train_transforms = transforms.Compose([
    transforms.Resize((PATCH, PATCH)),   # パッチ自体のサイズに揃える
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((PATCH, PATCH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ===========================
# Dataset クラス
# ===========================
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None, patch_size=384, num_patches=8):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.data = []
        self._prepare_data()

    def _prepare_data(self):
        for class_label in self.classes:
            class_path = os.path.join(self.root_dir, class_label)
            label_index = self.classes.index(class_label)
            if not os.path.isdir(class_path):
                continue

            for img_file in tqdm(os.listdir(class_path), desc=f'Loading {class_label}'):
                img_full_path = os.path.join(class_path, img_file)
                if os.path.isfile(img_full_path) and img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.data.append((img_full_path, label_index))

    def __len__(self):
        return len(self.data)

    def _random_crop(self, img: Image.Image):
        # img: PIL Image (RGB or L)
        W, H = img.size
        P = self.patch_size

        # 小さい場合は中央クロップ寄りにする（安全）
        if W < P or H < P:
            img = img.resize((max(W, P), max(H, P)))
            W, H = img.size

        left = random.randint(0, W - P)
        top  = random.randint(0, H - P)
        return img.crop((left, top, left + P, top + P))

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            # 2D白黒ならここは "L" が基本。timm使うので後で3ch化します。
            img = Image.open(img_path).convert("L")
        except:
            img = Image.new("L", (self.patch_size, self.patch_size))

        patches = []
        for _ in range(self.num_patches):
            p = self._random_crop(img)          # (P,P) を切り出し
            p = p.convert("RGB")                # 3ch化（timm pretrained前提）

            if self.transform:
                p = self.transform(p)           # (3,P,P)
            else:
                p = transforms.ToTensor()(p)

            patches.append(p)

        patches = torch.stack(patches, dim=0)   # (K,3,P,P)
        return patches, torch.tensor(label).float()
    

def main():
    set_seed()

    train_dataset = BreastCancerDataset(train_dir, classes, train_transforms, patch_size=224, num_patches=4)
    val_dataset   = BreastCancerDataset(val_dir, classes, val_test_transforms, patch_size=224, num_patches=4)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False,
                          num_workers=0, pin_memory=True)

# ===========================
# EfficientNet モデル
# ===========================
    class PatchMIL(nn.Module):
        def __init__(self, backbone_name="tf_efficientnet_b3_ns"):
            super().__init__()
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                num_classes=0,     # 特徴量
                global_pool="avg"
            )
            feat_dim = self.backbone.num_features
            self.head = nn.Linear(feat_dim, 1)  # binary logit

        def forward(self, x):
        # x: (B,K,3,P,P) or (B,3,P,P)
            if x.dim() == 4:
                x = x.unsqueeze(1)  # (B,1,3,P,P)

            B, K, C, H, W = x.shape
            x = x.view(B*K, C, H, W)
            feat = self.backbone(x)
            logit_patch = self.head(feat).view(B, K, 1)
            logit_img, _ = torch.max(logit_patch, dim=1)
            return logit_img
    
# device は最初に1回だけ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device,
      "cuda:", torch.cuda.is_available(),
      "gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

# モデル作成
    model = PatchMIL("tf_efficientnet_b3_ns")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

# ===========================
# BCEWithLogitsLoss (Sigmoid不要)
# ===========================
    optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===========================
# トレーニング
# ===========================
    num_epochs = 15
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    epoch_bar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in epoch_bar:
        # -------------------------------
        # Train
        # -------------------------------
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # -------------------------------
        # Validation
        # -------------------------------
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).view(-1, 1)

                outputs = model(inputs)                    # (B,1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        epoch_bar.set_postfix(
            loss=epoch_loss,
            train_acc=epoch_acc,
            val_loss=val_epoch_loss,
            val_acc=val_epoch_acc
        )

    # ===========================
    # 重み保存
    # ===========================
    output_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_michi"
    os.makedirs(output_dir, exist_ok=True)

    weight_dir = os.path.join(output_dir, "Weight")
    os.makedirs(weight_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{weight_dir}/breast_cancer_model.pth")

    # ===========================
    # 評価（AUCなど）
    # ===========================
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device).view(-1, 1)

            outputs = model(inputs).view(-1, 1)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    print("Accuracy:", acc)
    print("AUC:", auc)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)

    # ===========================
    # テストデータ推論
    # ===========================
    class TestDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
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
            if self.transform:
                image = self.transform(image)
            return image, fn


    test_dataset = TestDataset(test_dir, val_test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model.eval()
    predictions = []

    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Test Prediction"):
            images = images.to(device, non_blocking=True)
            outputs = model(images).view(-1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float().cpu().numpy().astype(int)

            for fn, p in zip(filenames, preds):
                image_id = os.path.splitext(fn)[0]
                predictions.append((image_id, p))

    prediction_dir = os.path.join(output_dir, "Prediction")
    os.makedirs(prediction_dir, exist_ok=True)

    submit_file_path = f"{prediction_dir}/sample_submit_michi_1210.csv"
    df = pd.DataFrame(predictions, columns=["image_id", "target"])
    df.to_csv(submit_file_path, index=False)

    print(f"サブミットファイルが {submit_file_path} に作成されました。")

    if __name__ == "__main__":
        main()