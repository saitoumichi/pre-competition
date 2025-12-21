"""
改善前後のモデルを比較するスクリプト
短縮版で両方のモデルを学習して性能を比較します
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
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

# 元のCNNモデル（改善前）
class OriginalCNNModel(nn.Module):
    def __init__(self):
        super(OriginalCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=100, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=100, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 28 * 28, 64)
        self.relu5 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.relu6 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 32)
        self.relu7 = nn.ReLU()
        self.out = nn.Linear(32, 1)  # バグ修正
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.pool3(self.relu4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.relu5(self.fc1(x)))
        x = self.dropout2(self.relu6(self.fc2(x)))
        x = self.relu7(self.fc3(x))
        x = self.sigmoid(self.out(x))
        return x

# 改善後のモデル（転移学習）
class ImprovedModel(nn.Module):
    def __init__(self):
        super(ImprovedModel, self).__init__()
        backbone = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        num_features = backbone.fc.in_features
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# データセットクラス
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
            for img_file in os.listdir(class_path):
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

def train_and_evaluate(model, train_loader, val_loader, model_name, num_epochs=3):
    """モデルを学習して評価する"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 改善前: BCELoss + Adam
    # 改善後: FocalLoss + AdamW
    if "Original" in model_name:
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        # Focal Loss
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
            def forward(self, inputs, targets):
                bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-bce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
                return focal_loss.mean()
        
        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 学習
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # 評価
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1).float()
            outputs = model(inputs)
            probs = outputs.cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    y_true = np.array(all_labels).flatten()
    y_probs = np.array(all_probs).flatten()
    y_pred = (y_probs >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    return {
        'accuracy': acc,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

def main():
    set_seed()
    
    # データパス
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "BreastCancer")
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "valid")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("データディレクトリが見つかりません。")
        print(f"train_dir: {train_dir}")
        print(f"val_dir: {val_dir}")
        return
    
    # データ拡張（改善前: シンプル、改善後: 強化）
    original_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    improved_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    classes = ["0", "1"]
    
    # データセット（改善前）
    print("改善前のモデルを学習中...")
    original_train_dataset = BreastCancerDataset(train_dir, classes, original_transforms)
    original_val_dataset = BreastCancerDataset(val_dir, classes, val_transforms)
    original_train_loader = DataLoader(original_train_dataset, batch_size=32, shuffle=True, num_workers=0)
    original_val_loader = DataLoader(original_val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    original_model = OriginalCNNModel()
    original_results = train_and_evaluate(original_model, original_train_loader, original_val_loader, "Original", num_epochs=3)
    
    # データセット（改善後）
    print("\n改善後のモデルを学習中...")
    improved_train_dataset = BreastCancerDataset(train_dir, classes, improved_transforms)
    improved_val_dataset = BreastCancerDataset(val_dir, classes, val_transforms)
    improved_train_loader = DataLoader(improved_train_dataset, batch_size=32, shuffle=True, num_workers=0)
    improved_val_loader = DataLoader(improved_val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    improved_model = ImprovedModel()
    improved_results = train_and_evaluate(improved_model, improved_train_loader, improved_val_loader, "Improved", num_epochs=3)
    
    # 結果の比較
    print("\n" + "="*60)
    print("性能比較結果")
    print("="*60)
    print(f"\n{'指標':<20} {'改善前':<15} {'改善後':<15} {'改善幅':<15}")
    print("-"*60)
    
    metrics = ['accuracy', 'auc', 'sensitivity', 'specificity']
    for metric in metrics:
        original_val = original_results[metric]
        improved_val = improved_results[metric]
        improvement = improved_val - original_val
        improvement_pct = (improvement / original_val * 100) if original_val > 0 else 0
        
        print(f"{metric.capitalize():<20} {original_val:<15.4f} {improved_val:<15.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")
    
    print("\n" + "="*60)
    print("結論:")
    print(f"Accuracy: {improved_results['accuracy'] - original_results['accuracy']:+.4f} ({((improved_results['accuracy'] - original_results['accuracy']) / original_results['accuracy'] * 100):+.1f}%)")
    print(f"AUC: {improved_results['auc'] - original_results['auc']:+.4f} ({((improved_results['auc'] - original_results['auc']) / original_results['auc'] * 100):+.1f}%)")
    print(f"Sensitivity (検出率): {improved_results['sensitivity'] - original_results['sensitivity']:+.4f} ({((improved_results['sensitivity'] - original_results['sensitivity']) / original_results['sensitivity'] * 100):+.1f}%)")
    print("="*60)

if __name__ == "__main__":
    main()

