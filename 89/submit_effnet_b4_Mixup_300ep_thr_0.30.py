import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import torch.optim as optim
from torch.optim import lr_scheduler # ★追加
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter

# ==========================================
# 1. 設定・準備
# ==========================================
def set_seed(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

# パス設定
train_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\train"
val_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\valid"
test_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\test"

# ★保存先を「200」に変えました
log_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_kazma\runs_effnet_b4_mixup_200"
output_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_kazma_effnet_b4_mixup_200"

os.makedirs(output_dir, exist_ok=True)
weight_dir = os.path.join(output_dir, "Weight")
os.makedirs(weight_dir, exist_ok=True)
prediction_dir = os.path.join(output_dir, "Prediction")
os.makedirs(prediction_dir, exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)
classes = ["0", "1"]

# ==========================================
# 2. 画像変換
# ==========================================
train_transforms = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.RandomCrop((380, 380)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. Dataset
# ==========================================
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
            if not os.path.isdir(class_path): continue
            for img_file in tqdm(os.listdir(class_path), desc=f'Loading {class_label}'):
                img_full_path = os.path.join(class_path, img_file)
                if os.path.isfile(img_full_path) and img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.data.append((img_full_path, label_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (380, 380))
        if self.transform:
            image = self.transform(image)
        return image, label

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

# ==========================================
# 4. DataLoader
# ==========================================
train_dataset = BreastCancerDataset(train_dir, classes, train_transforms)
val_dataset = BreastCancerDataset(val_dir, classes, val_test_transforms)

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 5. Model
# ==========================================
print("Creating model: tf_efficientnet_b4_ns")
model = timm.create_model("tf_efficientnet_b4_ns", pretrained=True, num_classes=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
prev_weight_path = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_kazma_effnet_b4_mixup\Weight\effnet_b4_mixup_model.pth"

if os.path.exists(prev_weight_path):
    print(f"★強くてニューゲーム！前回の重みをロードします: {prev_weight_path}")
    # 重みを上書きする
    # ※ strict=False は万が一の細かいエラーを無視してロードするおまじない
    model.load_state_dict(torch.load(prev_weight_path), strict=False)
else:
    print("★前回の重みが見つかりませんので、1から学習します。")
if torch.cuda.device_count() > 1:
    print(f"🔥 GPUを {torch.cuda.device_count()} 枚使ってフルパワー学習します！")
    model = nn.DataParallel(model)

# ==========================================
# 6. Loss, Optimizer & Scheduler
# ==========================================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# ★ここが300エポック用の最強設定
num_epochs = 300
# 学習率を徐々に下げて、最後に綺麗に着地させる（Cosine Annealing）
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# ==========================================
# 7. Training Loop
# ==========================================
alpha = 1.0 # Mixup強度

epoch_bar = tqdm(range(num_epochs), desc="Training Progress")

for epoch in epoch_bar:
    # --- Train (Mixup) ---
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.float().to(device).view(-1, 1)

        optimizer.zero_grad()

        # Mixup
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        index = torch.randperm(inputs.size(0)).to(device)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        outputs = model(mixed_inputs).view(-1, 1)
        loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # ★エポックの終わりに学習率を更新
    scheduler.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total

    # --- Validation ---
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

    writer.add_scalars("Loss", {"train": epoch_loss, "val": val_epoch_loss}, epoch)
    writer.add_scalars("Accuracy", {"train": epoch_acc, "val": val_epoch_acc}, epoch)

    # ログ表示（学習率 lr も表示するようにしました）
    current_lr = optimizer.param_groups[0]['lr']
    epoch_bar.set_postfix(loss=epoch_loss, val_acc=val_epoch_acc, lr=f"{current_lr:.6f}")

writer.close()

# 300エポック完走後のモデル（スケジューラーのおかげでこれが最強）
if isinstance(model, nn.DataParallel):
    torch.save(model.module.state_dict(), f"{weight_dir}/effnet_b4_mixup_300ep_model.pth")
else:
    torch.save(model.state_dict(), f"{weight_dir}/effnet_b4_mixup_300ep_model.pth")

# ==========================================
# 8. Evaluation
# ==========================================
print("\n=== 最適な閾値を探索中... ===")
model.eval()
all_labels, all_probs = [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.float().to(device).view(-1, 1)
        outputs = model(inputs).view(-1, 1)
        probs = torch.sigmoid(outputs)
        all_probs.extend(probs.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

y_true = np.array(all_labels)
y_prob = np.array(all_probs)

best_thr = 0.5
best_acc = 0.0
for thr in np.arange(0.1, 0.95, 0.05):
    y_pred_temp = (y_prob >= thr).astype(int)
    acc_temp = accuracy_score(y_true, y_pred_temp)
    if acc_temp > best_acc:
        best_acc = acc_temp
        best_thr = thr

print("-" * 30)
print(f"★200ep Mixup後の最強閾値: {best_thr:.2f}")
print(f"その時の検証正解率: {best_acc:.4f}")
print("-" * 30)

acc = accuracy_score(y_true, (y_prob >= best_thr).astype(int))
auc = roc_auc_score(y_true, y_prob)
print(f"Final Accuracy: {acc:.4f}")
print(f"Final AUC: {auc:.4f}")

# ==========================================
# 9. Test Prediction
# ==========================================
print(f"\n=== テストデータ推論開始 (TTAあり / 閾値: {best_thr:.2f}) ===")
test_dataset = TestDataset(test_dir, val_test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

predictions = []
with torch.no_grad():
    for images, filenames in tqdm(test_loader, desc="TTA Prediction"):
        images = images.to(device)
        output_normal = model(images)
        output_flip = model(torch.flip(images, [3]))
        outputs = (output_normal + output_flip) / 2
        probs = torch.sigmoid(outputs)
        preds = (probs >= best_thr).float().cpu().numpy().astype(int)
        for fn, p in zip(filenames, preds):
            image_id = os.path.splitext(fn)[0]
            predictions.append((image_id, p))

# 保存
submit_file_path = f"{prediction_dir}/submit_effnet_b4_Mixup_300ep_thr_{best_thr:.2f}.csv"
df = pd.DataFrame(predictions, columns=["image_id", "target"])
df.to_csv(submit_file_path, index=False)

print(f"完了！最強の300エポック版ファイルを作成しました: {submit_file_path}")
print(f"起きたら、このファイルを使ってアンサンブルしましょう！おやすみなさい！")