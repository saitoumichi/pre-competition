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
from torch.utils.tensorboard import SummaryWriter
#GPU 1枚使用 閾値 0.5

# 乱数固定
def set_seed(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

# データパス

train_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\train"
val_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\valid"
test_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\test"

log_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_kazma\runs_resnet50d"
writer = SummaryWriter(log_dir=log_dir)

classes = ["0", "1"]


from torchvision import transforms
import numpy as np # transforms.RandomAffine の 'degrees' に利用

# 画像変換（強化版）

train_transforms = transforms.Compose([
    # 1. リサイズとクロッピング（モデルの入力サイズに合わせる）
    transforms.Resize((256, 256)), # まず少し大きくリサイズ
    transforms.RandomCrop((224, 224)), # ランダムに224x224を切り出す (変更点: Scale Augmentationに近い効果)
    # 2. 幾何学的な変換 (Geometric Transforms)
    transforms.RandomHorizontalFlip(), # 水平反転 (既存)
    transforms.RandomRotation(degrees=30), # ランダムな回転（±15度以内） (追加)
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1), # 画像の最大10%分をランダムに平行移動 (追加)
        scale=(0.9, 1.1),     # 0.9倍から1.1倍にランダムに拡大縮小 (追加)
        shear=5               # シアー（せん断）変換（±5度以内） (追加)
    ),



    # 4. テンソル化と正規化（必須の最終ステップ）
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    # 検証・テストデータはリサイズのみで、ランダムな変換は行わない (既存)
    transforms.Resize((224, 224)),
    # 🚨 ここから下は非常に重要です！
    transforms.ToTensor(), # <--- これがないとPIL Imageのままになります
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

# ===========================
# Dataset クラス
# ===========================
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

            for img_file in tqdm(os.listdir(class_path), desc=f'Loading {class_label}'):
                img_full_path = os.path.join(class_path, img_file)
                if os.path.isfile(img_full_path) and img_file.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.data.append((img_full_path, label_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)
        return image, label


train_dataset = BreastCancerDataset(train_dir, classes, train_transforms)
val_dataset = BreastCancerDataset(val_dir, classes, val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ===========================
# EfficientNet モデル
# =========================== tf_efficientnet_b0_ns
model = timm.create_model(
    "resnet50d",
    pretrained=True,
    num_classes=1,   # 1出力（logit）
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ===========================
# BCEWithLogitsLoss (Sigmoid不要)
# ===========================
#criterion = nn.BCEWithLogitsLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)

pos_weight_value = 1569 / 806
pos_weight = torch.tensor([pos_weight_value], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# ===========================
# トレーニング
# ===========================

num_epochs = 13
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
        inputs = inputs.to(device)
        labels = labels.float().to(device).view(-1, 1)  # (batch,1)

        optimizer.zero_grad()

        outputs = model(inputs).view(-1, 1)  # 必ず (batch,1) にする
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
            inputs = inputs.to(device)
            labels = labels.float().to(device).view(-1, 1)

            outputs = model(inputs).view(-1, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) >= 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_epoch_loss = val_loss / len(val_loader.dataset)
    val_epoch_acc = val_correct / val_total

    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)

    # ===== TensorBoardに記録 =====
    writer.add_scalars("Loss", {
        "train": epoch_loss,
        "val": val_epoch_loss
    }, epoch)

    writer.add_scalars("Accuracy", {
        "train": epoch_acc,
        "val": val_epoch_acc
    }, epoch)



    epoch_bar.set_postfix(
        loss=epoch_loss,
        train_acc=epoch_acc,
        val_loss=val_epoch_loss,
        val_acc=val_epoch_acc
    )

writer.close()

# ===========================
# 重み保存
# ===========================
output_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_kazma_1215"
os.makedirs(output_dir, exist_ok=True)

weight_dir = os.path.join(output_dir, "Weight")
os.makedirs(weight_dir, exist_ok=True)

# DataParallelかどうかで保存方法を分ける
if isinstance(model, nn.DataParallel):
    state_dict = model.module.state_dict()
else:
    state_dict = model.state_dict()

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
        images = images.to(device)
        outputs = model(images).view(-1)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float().cpu().numpy().astype(int)

        for fn, p in zip(filenames, preds):
            image_id = os.path.splitext(fn)[0]
            predictions.append((image_id, p))

prediction_dir = os.path.join(output_dir, "Prediction")
os.makedirs(prediction_dir, exist_ok=True)

submit_file_path = f"{prediction_dir}/sample_submit_kazma_1211_GPU1_10.csv"
df = pd.DataFrame(predictions, columns=["image_id", "target"])
df.to_csv(submit_file_path, index=False)

print(f"サブミットファイルが {submit_file_path} に作成されました。")