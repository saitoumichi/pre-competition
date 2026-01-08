import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import warnings

# --- 安定化のための設定 ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # デバッグ用だがエラー回避に効くことがある
warnings.filterwarnings("ignore")

# cuDNN設定 (速度より安定性重視)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Config:
    # 既にB0, B1は終わったのでコメントアウトしています。
    # もしやり直したい場合はコメントを外してください。
    models_to_train = [
        # {'name': 'tf_efficientnetv2_b0', 'size': 224},
        # {'name': 'tf_efficientnetv2_b1', 'size': 240},
        {'name': 'tf_efficientnetv2_b2', 'size': 260},
        {'name': 'tf_efficientnetv2_b3', 'size': 300},
        {'name': 'tf_efficientnetv2_s',  'size': 380},
    ]
    
    # GPU 1枚で安定動作させるため、バッチサイズを調整
    # メモリ不足になる場合は 16 -> 8 に下げてください
    batch_size = 16 
    epochs = 20
    lr_backbone = 1e-4
    lr_head = 1e-3
    
    # パス設定
    base_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main"
    train_dir = os.path.join(base_dir, r"nakayamaken\BreastCancer\train")
    val_dir = os.path.join(base_dir, r"nakayamaken\BreastCancer\valid")
    
    save_dir = os.path.join(base_dir, "models_finetuned")
    os.makedirs(save_dir, exist_ok=True)

# データセット定義
class SingleModelDataset(Dataset):
    def __init__(self, root_dir, img_size, transform=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        self.data = []
        self.classes = ["0", "1"]
        
        if not os.path.exists(root_dir): return
        for cls in self.classes:
            c_path = os.path.join(root_dir, cls)
            if not os.path.isdir(c_path): continue
            label = self.classes.index(cls)
            for f in os.listdir(c_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append((os.path.join(c_path, f), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return img, torch.tensor(label, dtype=torch.float32).unsqueeze(0)

def get_transforms(img_size, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

def train_single_model(model_info, device):
    name = model_info['name']
    size = model_info['size']
    print(f"\n=========================================")
    print(f"Start Training: {name} (Size: {size})")
    print(f"=========================================")
    
    # 1. データセット
    train_ds = SingleModelDataset(Config.train_dir, size, transform=get_transforms(size, is_train=True))
    val_ds = SingleModelDataset(Config.val_dir, size, transform=get_transforms(size, is_train=False))
    
    # num_workers=0 でWindows特有のエラー回避 & 安定化
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 2. モデル作成
    model = timm.create_model(name, pretrained=True, num_classes=1)
    model = model.to(device)
    
    # ★重要: DataParallel は使わず、シングルGPUで確実に動かす
    
    # 3. Optimizer
    base_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n]

    optimizer = optim.Adam([
        {'params': base_params, 'lr': Config.lr_backbone},
        {'params': head_params, 'lr': Config.lr_head}
    ])
    
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # 4. 学習ループ
    best_acc = 0.0
    save_path = os.path.join(Config.save_dir, f"{name}_finetuned.pth")
    
    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0
        
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}"):
            # ★重要: メモリ配置を強制的に整列させる
            imgs = imgs.to(device).contiguous()
            lbls = lbls.to(device)
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 検証
        model.eval()
        preds_all, lbls_all = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device).contiguous()
                out = model(imgs)
                prob = torch.sigmoid(out).view(-1).cpu().numpy()
                preds_all.extend(prob)
                lbls_all.extend(lbls.view(-1).numpy())
        
        acc = accuracy_score(lbls_all, (np.array(preds_all) >= 0.5).astype(int))
        print(f"  Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Best Model Saved: {save_path}")
            
        scheduler.step(acc)
        
    print(f"Finished {name}. Best Acc: {best_acc:.4f}")
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()

def main():
    # GPU 0番のみ使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for info in Config.models_to_train:
        train_single_model(info, device)

if __name__ == "__main__":
    main()