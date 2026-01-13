import os
import sys
import numpy as np
import pandas as pd
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

import timm
from timm.data import Mixup
from timm.utils import ModelEmaV2
from timm.scheduler import CosineLRScheduler
from torchvision import transforms

# ==========================================
# CONFIG
# ==========================================
class CFG:
    seed = 42
    # ★ 768x768 の高解像度設定
    img_size = 768
    
    # ★ バッチサイズは2（メモリ対策）
    batch_size = 2  
    
    # ★ 勾配蓄積数：16回貯める（実質バッチサイズ = 2 * 16 = 32）
    accum_iter = 16
    
    num_workers = 4
    epochs = 50
    lr = 1e-4
    weight_decay = 0.05
    
    # ★ ConvNeXt Small (768ならBaseより強い)
    model_name = 'convnext_small.fb_in22k_ft_in1k_384'
    
    mixup_alpha = 0.8
    cutmix_alpha = 1.0
    mixup_prob = 0.2
    
    train_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer_mk2\train"
    val_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\valid"
    test_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\test"
    
    # ★ 出力先を変更（Baseの結果を消さないため）
    output_dir = r"D:\puresotu\workespace\result_refined_multi_gpu_768"
# ==========================================
# Utils & Dataset (変更なし)
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms(cfg):
    train_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
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
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (CFG.img_size, CFG.img_size))
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            filename = os.path.basename(img_path)
            return image, filename
        else:
            return image, torch.tensor(label).long()

# ==========================================
# ★ Training Function (ここを修正しました！)
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, mixup_fn, device, ema_model=None, accum_iter=1, start_update=0):
    model.train()
    running_loss = 0.0

    # 最初に勾配をリセット
    optimizer.zero_grad(set_to_none=True)

    # timm CosineLRScheduler は「累積update回数」で更新する
    update_step = start_update

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Train", leave=False)

    for step, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        with autocast(enabled=True):
            outputs = model(images)

            # BCEWithLogitsLoss: ラベルは float を想定
            if len(labels.shape) == 1:
                loss = criterion(outputs.view(-1), labels.float())
            else:
                # Mixup後は (B, 2) などになるので、positiveクラス(1)を使う
                loss = criterion(outputs.view(-1), labels[:, 1])

            # 勾配蓄積：平均化
            loss = loss / accum_iter

        scaler.scale(loss).backward()

        # accum_iter回に1回、または最後のバッチで必ず更新する
        do_update = ((step + 1) % accum_iter == 0) or ((step + 1) == len(loader))
        if do_update:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # EMA更新は DataParallel を考慮して module を渡す
            if ema_model is not None:
                ema_model.update(model.module if hasattr(model, "module") else model)

            if scheduler is not None:
                scheduler.step_update(update_step)
            update_step += 1

        # ログ表示用に元のスケールに戻して加算
        running_loss += (loss.item() * accum_iter) * images.size(0)
        pbar.set_postfix({'loss': float(loss.item() * accum_iter), 'updates': update_step})

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, update_step


# ==========================================
# Validation Function (変更なし)
# ==========================================
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
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    return epoch_loss, acc, auc, y_true, y_pred

# ==========================================
# Main Loop (2GPU対応)
# ==========================================
def main():
    set_seed(CFG.seed)
    os.makedirs(CFG.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    gpu_count = torch.cuda.device_count()
    print(f"Using device: {device}")
    print(f"Available GPUs: {gpu_count}") 

    train_transform, val_transform = get_transforms(CFG)

    train_dataset = BreastCancerDataset(CFG.train_dir, transform=train_transform)
    val_dataset = BreastCancerDataset(CFG.val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, 
                            num_workers=CFG.num_workers, pin_memory=True)

    mixup_fn = Mixup(
        mixup_alpha=CFG.mixup_alpha, cutmix_alpha=CFG.cutmix_alpha, 
        prob=CFG.mixup_prob, switch_prob=0.5, mode='batch',
        label_smoothing=0.05, num_classes=2
    )

    print(f"Creating model: {CFG.model_name}")
    print(f"Image Size: {CFG.img_size} x {CFG.img_size}")
    print(f"Batch Size: {CFG.batch_size} (Accum: {CFG.accum_iter} -> Effective: {CFG.batch_size * CFG.accum_iter})")

    model = timm.create_model(
        CFG.model_name,
        pretrained=True,
        num_classes=1,
        drop_path_rate=0.2,
    )
    model.to(device)

    # ==============================================
    # ★ここに追加！ 前回の続きから始めるコード
    # ==============================================
    # さっき保存された「last_model.pth」を読み込みます
    # ここに「コピーしたパス」をそのままドン！と貼ります
    resume_path = r"D:\puresotu\workespace\result_refined_multi_gpu_768\last_model.pth"
    
    if os.path.exists(resume_path):
        print(f"★ 前回の学習をロード中: {resume_path}")
        # DataParallelの 'module.' という文字を削除して読み込む
        state_dict = torch.load(resume_path)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("★ ロード成功！続きから学習します！")
    else:
        print("データが見つかりません。最初から学習します。")
    # ==============================================

    if gpu_count > 1:
        print(f"Wrapping model with DataParallel (GPUs: {gpu_count})")
        model = nn.DataParallel(model)
    
    if hasattr(model, 'module'):
        ema_model = ModelEmaV2(model.module, decay=0.999)
    else:
        ema_model = ModelEmaV2(model, decay=0.999)

    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    
    # 勾配蓄積後の「実際のoptimizer update回数」をt_initialにする
    updates_per_epoch = math.ceil(len(train_loader) / CFG.accum_iter)
    num_steps = int(CFG.epochs * updates_per_epoch)
    scheduler = CosineLRScheduler(
        optimizer, t_initial=num_steps, lr_min=1e-6,
        warmup_t=max(1, int(num_steps * 0.1)), warmup_lr_init=1e-6, cycle_limit=1
    )
    
    pos_weight = torch.tensor([2.0]).to(device) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)    
    scaler = GradScaler()

    best_auc = 0.0
    update_step = 0
    
    for epoch in range(CFG.epochs):
        print(f"\nEpoch {epoch+1}/{CFG.epochs}")
        
        # ★ 修正点3: accum_iter, update_step を渡す
        train_loss, update_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, mixup_fn, device, ema_model,
            accum_iter=CFG.accum_iter, start_update=update_step
        )
        
        # Validation
        val_loss, val_acc, val_auc, _, _ = validate(model, val_loader, criterion, device)
        
        # EMA Validation
        ema_loss, ema_acc, ema_auc, y_true, y_pred = validate(ema_model.module, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val  [Raw] Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")
        print(f"Val  [EMA] Loss: {ema_loss:.4f} Acc: {ema_acc:.4f} AUC: {ema_auc:.4f}")
        
        if ema_auc > best_auc:
            best_auc = ema_auc
            save_path = os.path.join(CFG.output_dir, "best_model.pth")
            torch.save(ema_model.module.state_dict(), save_path)
            print(f"Best Model Saved! (AUC: {best_auc:.4f})")
            
            cm = confusion_matrix(y_true, y_pred)
            print("Confusion Matrix:\n", cm)

    last_save_path = os.path.join(CFG.output_dir, "last_model.pth")
    torch.save(ema_model.module.state_dict(), last_save_path)
    print(f"Last Model Saved to {last_save_path}")

    # ==========================================
    # Inference (TTA)
    # ==========================================
    print("\nStarting Inference with TTA...")
    
    model_infer = timm.create_model(CFG.model_name, pretrained=False, num_classes=1)
    model_infer.load_state_dict(torch.load(os.path.join(CFG.output_dir, "best_model.pth")))
    model_infer.to(device)
    model_infer.eval()
    
    if gpu_count > 1:
         model_infer = nn.DataParallel(model_infer)

    test_dataset = BreastCancerDataset(CFG.test_dir, is_test=True, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
    
    predictions = []
    
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Prediction"):
            images = images.to(device)
            
            # TTA: Original + Horizontal Flip
            out1 = model_infer(images).view(-1)
            prob1 = torch.sigmoid(out1)
            
            images_flipped = torch.flip(images, dims=[3])
            out2 = model_infer(images_flipped).view(-1)
            prob2 = torch.sigmoid(out2)
            
            # ★ 追加TTA: Vertical Flip (上下反転)
            images_v = torch.flip(images, dims=[2])
            out3 = model_infer(images_v).view(-1)
            prob3 = torch.sigmoid(out3)

            # 3つの平均をとる
            avg_prob = (prob1 + prob2 + prob3) / 3.0
            preds = (avg_prob >= 0.5).int().cpu().numpy()
            
            for fn, p in zip(filenames, preds):
                image_id = os.path.splitext(fn)[0]
                predictions.append((image_id, p))
                
    df = pd.DataFrame(predictions, columns=["image_id", "target"])
    sub_path = os.path.join(CFG.output_dir, "submission_refined_michi_gpu.csv")        
    df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

if __name__ == "__main__":
    main()