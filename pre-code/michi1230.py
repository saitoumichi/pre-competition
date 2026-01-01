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
from timm.data import Mixup
from timm.utils import ModelEmaV2
from timm.scheduler import CosineLRScheduler
from torchvision import transforms

# ==========================================
# CONFIG
# ==========================================
class CFG:
    seed = 42
    img_size = 384
    
    # 2GPUならメモリに余裕が出るのでバッチサイズを増やせます
    # 16 -> 32 に変更 (GPUメモリ不足なら減らしてください)
    batch_size = 32 # <--- 2GPU対応: バッチサイズ倍増推奨
    
    num_workers = 4
    epochs = 30
    lr = 5e-4
    weight_decay = 0.05
    
    model_name = 'convnext_tiny.fb_in22k_ft_in1k_384'
    
    mixup_alpha = 0.8
    cutmix_alpha = 1.0
    mixup_prob = 0.0
    
    train_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer_mk2\train"
    val_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\valid"
    test_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\test"
    output_dir = r"./result_refined_multi_gpu" # 出力先変更

# ==========================================
# Utils & Dataset (変更なし)
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
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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
# Training Function (変更なし)
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, mixup_fn, device, ema_model=None):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)
            
        optimizer.zero_grad()
        with autocast(enabled=True):
            outputs = model(images)
            if len(labels.shape) == 1:
                loss = criterion(outputs.view(-1), labels.float())
            else:
                loss = criterion(outputs.view(-1), labels[:, 1])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if ema_model:
            # DataParallelの場合、model.module を渡す必要があるが
            # ModelEmaV2 は内部でうまく処理してくれる場合もある。
            # 安全のため、DataParallelなら .module を渡すようにMainで設定する。
            ema_model.update(model)
            
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({'loss': loss.item()})
        
    scheduler.step_update(num_updates=len(loader))
    return running_loss / len(loader.dataset)

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
    
    # GPUの枚数確認
    gpu_count = torch.cuda.device_count()
    print(f"Using device: {device}")
    print(f"Available GPUs: {gpu_count}") # <--- 確認用

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
    model = timm.create_model(
        CFG.model_name,
        pretrained=True,
        num_classes=1,
        drop_path_rate=0.2,
    )
    model.to(device)

    # ========================================================
    # 2GPU対応: DataParallel でラップ
    # ========================================================
    if gpu_count > 1:
        print(f"Wrapping model with DataParallel (GPUs: {gpu_count})")
        model = nn.DataParallel(model) # <--- これだけで複数GPUに分散されます
    
    # EMAの設定: DataParallelを使っている場合は .module を取り出して渡すのが安全
    if hasattr(model, 'module'):
        ema_model = ModelEmaV2(model.module, decay=0.999)
    else:
        ema_model = ModelEmaV2(model, decay=0.999)

    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    
    num_steps = int(CFG.epochs * len(train_loader))
    scheduler = CosineLRScheduler(
        optimizer, t_initial=num_steps, lr_min=1e-6, 
        warmup_t=int(num_steps*0.1), warmup_lr_init=1e-6, cycle_limit=1
    )
    
    # 「1」のデータを重視するように重みを設定 (例: 5倍〜10倍)
    pos_weight = torch.tensor([2.0]).to(device) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)    
    scaler = GradScaler()

    best_acc = 0.0
    
    for epoch in range(CFG.epochs):
        print(f"\nEpoch {epoch+1}/{CFG.epochs}")
        
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, mixup_fn, device, ema_model
        )
        
        # Validation
        val_loss, val_acc, val_auc, _, _ = validate(model, val_loader, criterion, device)
        
        # EMA Validation
        ema_loss, ema_acc, ema_auc, y_true, y_pred = validate(ema_model.module, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val  [Raw] Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")
        print(f"Val  [EMA] Loss: {ema_loss:.4f} Acc: {ema_acc:.4f} AUC: {ema_auc:.4f}")
        
        # AUCでベストを更新するかチェック
        if ema_auc > best_acc:  # 変数名はbest_accのままでOK（中身はAUCとして使う）
            best_acc = ema_auc  # AUCを記録
            save_path = os.path.join(CFG.output_dir, "best_model.pth")
            torch.save(ema_model.module.state_dict(), save_path)
            print(f"Best Model Saved! (AUC: {best_acc:.4f})") # 表示もAUCに変更
            
            cm = confusion_matrix(y_true, y_pred)
            print("Confusion Matrix:\n", cm)

    # 【追加】ループ終了後に、最後のモデルも念のため保存しておく
    last_save_path = os.path.join(CFG.output_dir, "last_model.pth")
    torch.save(ema_model.module.state_dict(), last_save_path)
    print(f"Last Model Saved to {last_save_path}")

    # ==========================================
    # Inference (TTA)
    # ==========================================
    print("\nStarting Inference with TTA...")
    
    # モデルのロード (保存時に皮を剥いでいるので、通常通りロード可能)
    # 推論時も2GPU使いたいなら再度DataParallelしても良いが、
    # 複雑さを避けるためここではシングルGPUまたはロード後に再ラップする
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
            
            out1 = model_infer(images).view(-1)
            prob1 = torch.sigmoid(out1)
            
            images_flipped = torch.flip(images, dims=[3])
            out2 = model_infer(images_flipped).view(-1)
            prob2 = torch.sigmoid(out2)
            
            avg_prob = (prob1 + prob2) / 2.0
            preds = (avg_prob >= 0.5).int().cpu().numpy()
            
            for fn, p in zip(filenames, preds):
                image_id = os.path.splitext(fn)[0]
                predictions.append((image_id, p))
                
    df = pd.DataFrame(predictions, columns=["image_id", "target"])
    sub_path = os.path.join(CFG.output_dir, "submission_refined_multi_gpu.csv")
    df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

if __name__ == "__main__":
    main()