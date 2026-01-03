import os
import sys
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import timm
from timm.data import Mixup
from timm.scheduler import CosineLRScheduler
from torchvision import transforms

# ==========================================
# CONFIG: 画質特化型 (EffNetV2-S / 384px)
# ==========================================
class CFG:
    seed = 42
    
    # ★ここが勝負の分かれ目！画質を大幅アップ
    img_size = 384 
    
    # 画質を上げた分、バッチサイズは小さくしないと落ちます
    batch_size = 32  # もし落ちたら 4 にしてください
    
    num_workers = 2
    epochs = 100     # V2は学習が早いので15で十分収束します
    lr = 1e-4       # ファインチューニングなので控えめに
    weight_decay = 1e-4
    
    # ★最新モデル: EfficientNetV2 Small
    # in21k_ft_in1k: 2100万枚の画像で事前学習した「超・博識」な重みです
    model_name = 'tf_efficientnetv2_s.in21k_ft_in1k'
    
    mixup_alpha = 0.8
    cutmix_alpha = 1.0
    mixup_prob = 1.0
    
    base_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken"
    train_dir = os.path.join(base_dir, r"BreastCancer_mk2\train")
    val_dir   = os.path.join(base_dir, r"BreastCancer\valid")
    test_dir  = os.path.join(base_dir, r"BreastCancer\test")
    output_dir = r"./result_effnetv2_384_100ep"

# ==========================================
# Functions
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_transforms(cfg):
    train_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # EffNetV2推奨の正規化
    ])
    val_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return train_transform, val_transform

class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, classes=["0", "1"], transform=None, is_test=False):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.is_test = is_test
        self.data = []
        if self.is_test:
            files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for f in files: self.data.append((os.path.join(root_dir, f), -1))
        else:
            for class_label in self.classes:
                class_path = os.path.join(self.root_dir, class_label)
                if not os.path.isdir(class_path): continue
                label_idx = self.classes.index(class_label)
                for f in os.listdir(class_path):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.data.append((os.path.join(class_path, f), label_idx))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        if self.is_test: return image, os.path.basename(img_path)
        return image, torch.tensor(label).long()

# ==========================================
# Main
# ==========================================
def main():
    set_seed(CFG.seed)
    os.makedirs(CFG.output_dir, exist_ok=True)
    device = torch.device("cuda")
    print(f"Model: {CFG.model_name} | Size: {CFG.img_size} (高画質モード)")

    # Data
    train_tf, val_tf = get_transforms(CFG)
    train_ds = BreastCancerDataset(CFG.train_dir, transform=train_tf)
    val_ds = BreastCancerDataset(CFG.val_dir, transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

    # Mixup
    mixup_fn = Mixup(mixup_alpha=CFG.mixup_alpha, cutmix_alpha=CFG.cutmix_alpha, prob=CFG.mixup_prob, switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=2)

    # Model
    model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1, drop_path_rate=0.2)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    scheduler = CosineLRScheduler(optimizer, t_initial=CFG.epochs, lr_min=1e-6, warmup_t=3, warmup_lr_init=1e-6)

    best_auc = 0.0

    # Train Loop
    for epoch in range(CFG.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.epochs}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, labels = mixup_fn(imgs, labels)
            
            optimizer.zero_grad()
            with autocast(enabled=True):
                output = model(imgs).view(-1)
                loss = criterion(output, labels[:, 1])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        scheduler.step(epoch + 1)
        
        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs).view(-1)
                preds.extend(torch.sigmoid(output).cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        auc = roc_auc_score(targets, preds)
        print(f"Epoch {epoch+1} Val AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(CFG.output_dir, "best_effnetv2.pth"))
            print(f"Best Model Saved! AUC: {best_auc:.4f}")

    # Inference (384pxで推論)
    print("\nInference...")
    model.load_state_dict(torch.load(os.path.join(CFG.output_dir, "best_effnetv2.pth")))
    model.eval()
    
    test_ds = BreastCancerDataset(CFG.test_dir, is_test=True, transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
    
    results = []
    with torch.no_grad():
        for imgs, fnames in tqdm(test_loader):
            imgs = imgs.to(device)
            output = torch.sigmoid(model(imgs).view(-1))
            for f, p in zip(fnames, output.cpu().numpy()):
                results.append((os.path.splitext(f)[0], (p >= 0.5).astype(int)))
    
    df = pd.DataFrame(results, columns=["image_id", "target"])
    df.to_csv(os.path.join(CFG.output_dir, "submission_effnetv2_384.csv"), index=False)
    print("EfficientNetV2 (384px) submission saved!")

if __name__ == "__main__":
    main()