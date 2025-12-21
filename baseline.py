import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, efficientnet_b3, densenet121
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve

def set_seed(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

# データディレクトリのパスを設定（実際のパスに合わせて変更してください）
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "BreastCancer")
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "valid")
test_dir = os.path.join(data_dir, "test")

classes = ["0", "1"]

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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
                if os.path.isfile(img_full_path) and img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.data.append((img_full_path, label_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = BreastCancerDataset(root_dir=train_dir, classes=classes, transform=train_transforms)
val_dataset = BreastCancerDataset(root_dir=val_dir, classes=classes, transform=val_test_transforms)

batch_size = 64
num_workers = 0
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Focal Lossの実装（クラス不均衡に対応）
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 転移学習ベースのモデル（ResNet50を使用）
class BreastCancerModel(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super(BreastCancerModel, self).__init__()
        self.model_name = model_name
        
        if model_name == 'resnet50':
            backbone = resnet50(pretrained=pretrained)
            # 最後の全結合層を削除
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            num_features = backbone.fc.in_features
        elif model_name == 'efficientnet_b3':
            backbone = efficientnet_b3(pretrained=pretrained)
            self.features = backbone.features
            num_features = backbone.classifier[1].in_features
        elif model_name == 'densenet121':
            backbone = densenet121(pretrained=pretrained)
            self.features = backbone.features
            num_features = backbone.classifier.in_features
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # カスタム分類ヘッド
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        if self.model_name == 'resnet50':
            x = self.features(x)
            x = x.view(x.size(0), -1)
        elif self.model_name == 'efficientnet_b3':
            x = self.features(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
        elif self.model_name == 'densenet121':
            x = self.features(x)
            x = nn.functional.relu(x, inplace=True)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        return x

# 転移学習モデルの作成（ResNet50を使用、EfficientNetやDenseNetにも変更可能）
model = BreastCancerModel(model_name='resnet50', pretrained=True)
print(model)

# Focal Lossを使用（クラス不均衡に対応、通常のBCELossに戻す場合は nn.BCELoss() を使用）
criterion = FocalLoss(alpha=1, gamma=2)

# 学習率を調整（転移学習では初期は小さめに）
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# 学習率スケジューラーを追加
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

num_epochs = 20  # エポック数を増やす（転移学習ではより多くのエポックが有効）
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epoch_bar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")

for epoch in epoch_bar:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.view(-1, 1).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            preds = (outputs >= 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    val_epoch_acc = val_correct / val_total
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)
    
    # 学習率スケジューラーを更新
    scheduler.step(val_epoch_loss)
    
    epoch_bar.set_postfix(loss=epoch_loss, train_acc=epoch_acc, val_loss=val_epoch_loss, val_acc=val_epoch_acc)

output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)
weight_dir = os.path.join(output_dir, 'Weight')
os.makedirs(weight_dir, exist_ok=True)

torch.save(model.state_dict(), f'{weight_dir}/breast_cancer_model.pth')

weight_path = os.path.join(weight_dir, 'breast_cancer_model.pth')
model.load_state_dict(torch.load(weight_path))
model.eval()

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.view(-1, 1).float()
        outputs = model(inputs)
        probs_tensor = outputs
        probs = probs_tensor.cpu().numpy().flatten()
        all_probs.extend(probs)
        preds = (outputs >= 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

y_true = np.array(all_labels).flatten()
y_pred = np.array(all_preds).flatten()
y_probs = np.array(all_probs).flatten()

# 閾値の最適化（ROC曲線から最適な閾値を決定）
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
optimal_idx = np.argmax(tpr - fpr)  # Youden's J統計量を使用
optimal_threshold = thresholds[optimal_idx]

print(f"\n=== 閾値の最適化結果 ===")
print(f"デフォルト閾値 (0.5) での評価:")
cm_default = confusion_matrix(y_true, y_pred)
acc_default = accuracy_score(y_true, y_pred)
TN_default, FP_default, FN_default, TP_default = cm_default.ravel()
sensitivity_default = TP_default / (TP_default + FN_default) if (TP_default + FN_default) > 0 else 0
specificity_default = TN_default / (TN_default + FP_default) if (TN_default + FP_default) > 0 else 0

print(f"  Accuracy: {acc_default:.4f}")
print(f"  Sensitivity (Recall): {sensitivity_default:.4f}")
print(f"  Specificity: {specificity_default:.4f}")

# 最適な閾値での評価
y_pred_optimal = (y_probs >= optimal_threshold).astype(int)
cm_optimal = confusion_matrix(y_true, y_pred_optimal)
acc_optimal = accuracy_score(y_true, y_pred_optimal)
TN_optimal, FP_optimal, FN_optimal, TP_optimal = cm_optimal.ravel()
sensitivity_optimal = TP_optimal / (TP_optimal + FN_optimal) if (TP_optimal + FN_optimal) > 0 else 0
specificity_optimal = TN_optimal / (TN_optimal + FP_optimal) if (TN_optimal + FP_optimal) > 0 else 0

print(f"\n最適閾値 ({optimal_threshold:.4f}) での評価:")
print(f"  Accuracy: {acc_optimal:.4f}")
print(f"  Sensitivity (Recall): {sensitivity_optimal:.4f}")
print(f"  Specificity: {specificity_optimal:.4f}")

# AUCの計算
auc = roc_auc_score(y_true, y_probs)
print(f"\nAUC: {auc:.4f}")

# 最適な閾値を保存（テストデータの予測で使用）
print(f"\n最適な閾値 {optimal_threshold:.4f} を使用してテストデータを予測します。")

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [
            f for f in sorted(os.listdir(self.root_dir))
            if os.path.isfile(os.path.join(self.root_dir, f))
            and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
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

test_dataset = TestDataset(root_dir=test_dir, transform=val_test_transforms)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0) #batch = 64

model.eval()
predictions = []
with torch.no_grad():
    for images, filenames in tqdm(test_loader, desc="Test Prediction"):
        images = images.to(device)
        outputs = model(images)
        # 最適な閾値を使用（検証データで決定した閾値）
        preds = (outputs.squeeze().cpu().numpy() >= optimal_threshold).astype(int)
        for fn, p in zip(filenames, preds):
            image_id = os.path.splitext(fn)[0]
            predictions.append((image_id, p))

prediction_dir = os.path.join(output_dir, 'Prediction')
os.makedirs(prediction_dir, exist_ok=True)

submit_file_path = f'{prediction_dir}/sample_submit.csv'
df = pd.DataFrame(predictions, columns=['image_id', 'target'])
df.to_csv(submit_file_path, index=False)
print(f"サブミットファイルが {submit_file_path} に作成されました。")
df.head()
