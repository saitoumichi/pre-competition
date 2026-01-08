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
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

import timm
from timm.scheduler import CosineLRScheduler
from torchvision import transforms

# ==========================================
# CONFIG: 追加学習 100epoch (Base V2 512px)
# ==========================================
class CFG:
    seed = 42
    
    img_size = 512
    batch_size = 8
    accum_iter = 4  # 実質バッチサイズ = 32
    
    num_workers = 0 
    
    # ★変更: じっくり100回
    epochs = 100     
    
    # ★変更: 仕上げなので学習率を1/10に下げる
    lr = 1e-5       
    weight_decay = 1e-4
    
    # ★重要: 前回成功したモデル名に合わせる (Base V2)
    model_name = 'convnextv2_base.fcmae_ft_in22k_in1k'
    
    use_mixup = False 
    
    base_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken"
    train_dir = os.path.join(base_dir, r"BreastCancer\train")
    val_dir   = os.path.join(base_dir, r"BreastCancer\valid")
    test_dir  = os.path.join(base_dir, r"BreastCancer\test")
    
    # ★変更: 新しい保存先 (100ep用)
    output_dir = r"D:\puresotu\workespace\result_refined_multi_gpu_small_v2_100ep"
    
    # ★追加: 前回のベストモデルのパス (ここから再開)
    # 前回の output_dir にある best_model.pth を指定
    resume_path = r"D:\puresotu\workespace\result_refined_multi_gpu_small_v2\best_model.pth"

# ==========================================
# Utils & Dataset
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms(cfg):
    train_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15), 
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
            for f in files:
                self.data.append((os.path.join(root_dir, f), -1))
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
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (CFG.img_size, CFG.img_size))
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, os.path.basename(img_path)
        else:
            return image, torch.tensor(label).long()

# ==========================================
# Training Function
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, accum_iter=1):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Train", leave=False)
    
    for step, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        with autocast(enabled=True):
            outputs = model(images).view(-1)
            loss = criterion(outputs, labels.float())
            loss = loss / accum_iter

        scaler.scale(loss).backward()
        
        if (step + 1) % accum_iter == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        running_loss += (loss.item() * accum_iter) * images.size(0)
        pbar.set_postfix({'loss': loss.item() * accum_iter})
        
    scheduler.step_update(num_updates=len(loader))
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Valid", leave=False):
            images = images.to(device)
            labels = labels.to(device).float()
            
            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs)
            
            preds_list.extend(probs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    y_true = np.array(labels_list)
    y_prob = np.array(preds_list)
    
    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0.5
    except:
        auc = 0.5
        
    best_acc = 0.0
    best_thr = 0.5
    for thr in np.arange(0.1, 0.9, 0.05):
        acc = accuracy_score(y_true, (y_prob >= thr).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
            
    return epoch_loss, best_acc, auc, best_thr

# ==========================================
# Main Loop
# ==========================================
def main():
    set_seed(CFG.seed)
    os.makedirs(CFG.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data Check
    if not os.path.exists(CFG.train_dir):
        print(f"エラー: 訓練データが見つかりません -> {CFG.train_dir}")
        return

    train_transform, val_transform = get_transforms(CFG)
    train_dataset = BreastCancerDataset(CFG.train_dir, transform=train_transform)
    val_dataset = BreastCancerDataset(CFG.val_dir, transform=val_transform)
    
    all_labels = [label for _, label in train_dataset.data]
    num_neg = all_labels.count(0)
    num_pos = all_labels.count(1)
    pos_weight_val = (num_neg / num_pos) * 1.2
    print(f"Data Stats: Neg={num_neg}, Pos={num_pos} -> Pos Weight={pos_weight_val:.4f}")
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, 
                            num_workers=CFG.num_workers, pin_memory=True)

    print(f"Creating model: {CFG.model_name} (Size: {CFG.img_size})")
    model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1, drop_path_rate=0.2)
    model.to(device)
    
    # ★ Resume Logic
    if os.path.exists(CFG.resume_path):
        print(f"============== Resume from: {CFG.resume_path} ==============")
        try:
            state_dict = torch.load(CFG.resume_path, map_location=device)
            # DDPのmodule.接頭辞対策
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=True)
            print(">> Weights loaded successfully!")
        except Exception as e:
            print(f"!! Warning: Load failed ({e}). Starting from scratch.")
    else:
        print(f"!! Warning: Resume file not found at {CFG.resume_path}")
        print("Starting from scratch (ImageNet weights).")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    
    num_steps = int(CFG.epochs * len(train_loader))
    scheduler = CosineLRScheduler(optimizer, t_initial=num_steps, lr_min=1e-7, warmup_t=int(num_steps*0.05), warmup_lr_init=1e-7)
    
    pos_weight_tensor = torch.tensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)    
    scaler = GradScaler()

    best_score = 0.0 
    
    for epoch in range(CFG.epochs):
        print(f"\nEpoch {epoch+1}/{CFG.epochs}")
        
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, device, 
            accum_iter=CFG.accum_iter
        )
        
        val_loss, val_acc, val_auc, val_thr = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val AUC: {val_auc:.4f} | Acc: {val_acc:.4f} (Thr: {val_thr:.2f})")
        
        if val_auc > best_score:
            best_score = val_auc
            save_path = os.path.join(CFG.output_dir, "best_model.pth")
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"★ Best Model Updated! (AUC: {best_score:.4f})")

    # ==========================================
    # Inference
    # ==========================================
    print("\nStarting Inference...")
    
    model = timm.create_model(CFG.model_name, pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(os.path.join(CFG.output_dir, "best_model.pth")))
    model.to(device)
    model.eval()
    
    test_dataset = BreastCancerDataset(CFG.test_dir, is_test=True, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
    
    predictions = []
    
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Prediction"):
            images = images.to(device)
            out1 = model(images).view(-1)
            out2 = model(torch.flip(images, dims=[3])).view(-1)
            prob1 = torch.sigmoid(out1)
            prob2 = torch.sigmoid(out2)
            avg_prob = (prob1 + prob2) / 2.0
            
            # アンサンブル用に確率(prob)も保存したいので、今回はprobもCSVに残す仕様にします
            # ここではシンプルに予測ラベル(0/1)と確率(prob)両方持ったCSVにしてもいいですが、
            # 提出フォーマットに合わせて 0/1 にします。
            preds = (avg_prob >= 0.5).int().cpu().numpy()
            probs_np = avg_prob.cpu().numpy()
            
            for fn, p, prob_val in zip(filenames, preds, probs_np):
                image_id = os.path.splitext(fn)[0]
                # list: [image_id, target, prob]
                predictions.append([image_id, p, prob_val])
                
    # targetとprobの両方を保存しておくと後でSoft Votingに使えて便利
    df = pd.DataFrame(predictions, columns=["image_id", "target", "prob"])
    
    # 提出用は target のみが必要なので整形して保存
    df_submit = df[["image_id", "target"]]
    sub_path = os.path.join(CFG.output_dir, "submission_100ep.csv")        
    df_submit.to_csv(sub_path, index=False)
    
    # 確率版も保存（アンサンブル用）
    prob_path = os.path.join(CFG.output_dir, "submission_100ep_prob.csv")
    df.to_csv(prob_path, index=False)
    
    print(f"完了！提出ファイル: {sub_path}")
    print(f"確率ファイル（アンサンブル用）: {prob_path}")

if __name__ == "__main__":
    main()