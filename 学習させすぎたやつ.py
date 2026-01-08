import os
import sys
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import timm
from timm.utils import ModelEmaV2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
import random
import socket

# ★★★ Windowsエラー回避設定 ★★★
os.environ["USE_LIBUV"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

# 警告抑制
warnings.filterwarnings("ignore")

# ==========================================
# 設定 (Configuration)
# ==========================================
class Config:
    seed = 42
    
    img_size = 640
    num_classes = 1
    
    # モデル設定
    model_name = "convnextv2_base.fcmae_ft_in22k_in1k"
    
    # 学習設定
    batch_size = 4        
    accum_steps = 4       
    
    epochs = 50           
    lr = 2e-4             
    min_lr = 1e-6
    weight_decay = 0.05   
    max_grad_norm = 1.0 
    
    # MixUp有効化 (手動実装版)
    use_mixup = True      
    mixup_alpha = 0.8
    mixup_prob = 0.5      
    
    # EMA設定
    use_ema = True
    ema_decay = 0.999     
    
    # パス
    train_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\train"
    val_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\valid"
    test_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\test"
    output_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_final_improved"

# ==========================================
# Utils
# ==========================================
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def set_seed(seed=1234, rank=0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_main_process(rank):
    return rank == 0

# ==========================================
# Manual Binary Mixup
# ==========================================
def binary_mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0)).to(data.device)
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    mixed_data = lam * data + (1 - lam) * shuffled_data
    
    # ターゲットもブレンド
    mixed_targets = lam * targets + (1 - lam) * shuffled_targets
    
    return mixed_data, mixed_targets

# ==========================================
# GeM Pooling
# ==========================================
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

# ==========================================
# Model
# ==========================================
class BreastCancerModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")
        num_features = self.backbone.num_features
        self.global_pool = GeM()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        logits = self.head(x)
        return logits

# ==========================================
# Dataset & Transforms (Corrected)
# ==========================================
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, classes=None, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.data = []
        
        if not is_test:
            self.classes = classes
            for class_label in self.classes:
                class_path = os.path.join(self.root_dir, class_label)
                if not os.path.isdir(class_path): continue
                label_index = self.classes.index(class_label)
                files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for f in files:
                    self.data.append((os.path.join(class_path, f), label_index))
        else:
            files = sorted([f for f in os.listdir(self.root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for f in files:
                self.data.append((os.path.join(self.root_dir, f), f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, target = self.data[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((Config.img_size, Config.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        if self.is_test:
            return img, target
        else:
            return img, torch.tensor(target, dtype=torch.float32)

def get_transforms():
    return {
        "train": A.Compose([
            # size=(h, w) で指定 (RandomResizedCrop)
            A.RandomResizedCrop(size=(Config.img_size, Config.img_size), scale=(0.85, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.3),
            A.CoarseDropout(max_holes=8, max_height=Config.img_size//20, max_width=Config.img_size//20, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]),
        "valid": A.Compose([
            # height=, width= で指定 (Resize) 
            A.Resize(height=Config.img_size, width=Config.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    }

# ==========================================
# Validation Function
# ==========================================
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1))
            
            running_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs).squeeze()
            if probs.ndim == 0: probs = probs.unsqueeze(0)
            if torch.isnan(probs).any():
                probs = torch.nan_to_num(probs, nan=0.0)
            
            all_probs.append(probs)
            all_labels.append(labels)
            
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    dataset_len = len(loader.dataset)
    if dataset_len == 0:
        return 0.0, 0.0, 0.0, 0.5, [], []

    epoch_loss = running_loss / dataset_len
    
    y_true = all_labels.cpu().numpy()
    y_prob = all_probs.cpu().numpy()
    
    if len(np.unique(y_true)) > 1:
        val_auc = roc_auc_score(y_true, y_prob)
    else:
        val_auc = 0.0
        
    best_acc = 0.0
    best_thr = 0.5
    for thr in np.arange(0.05, 0.95, 0.05):
        y_pred_tmp = (y_prob >= thr).astype(int)
        acc_tmp = accuracy_score(y_true, y_pred_tmp)
        if acc_tmp > best_acc:
            best_acc = acc_tmp
            best_thr = thr
            
    return epoch_loss, best_acc, val_auc, best_thr, y_true, y_prob

# ==========================================
# Main Worker
# ==========================================
def main_worker(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ["USE_LIBUV"] = "0"
    
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_seed(Config.seed, rank)
    
    transforms_dict = get_transforms()
    train_dataset = BreastCancerDataset(Config.train_dir, classes=["0", "1"], transform=transforms_dict["train"])
    val_dataset = BreastCancerDataset(Config.val_dir, classes=["0", "1"], transform=transforms_dict["valid"])
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, sampler=val_sampler, num_workers=0, pin_memory=True, drop_last=False)
    
    model = BreastCancerModel(Config.model_name).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    model_ema = None
    if Config.use_ema:
        model_ema = ModelEmaV2(model, decay=Config.ema_decay, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.epochs, T_mult=1, eta_min=Config.min_lr
    )
    
    # Mixup時はBCE、それ以外も今回はBCEで統一
    criterion = nn.BCEWithLogitsLoss()
    
    scaler = torch.cuda.amp.GradScaler()
    
    best_auc_score = 0.0
    final_best_thr = 0.5
    
    if is_main_process(rank):
        print(f"Start Training: {Config.model_name} on {world_size} GPUs")
        print(f"  - Mixup: {Config.use_mixup} (Manual)")
        print(f"  - EMA: {Config.use_ema}")
        print(f"  - Augmentation: Enhanced")
    
    for epoch in range(Config.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        train_loss_sum = torch.tensor(0.0).to(device)
        total_batches = 0
        
        if is_main_process(rank):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}", leave=False)
        else:
            pbar = train_loader
            
        optimizer.zero_grad()
        
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device).view(-1, 1)
            
            # 手動Mixup適用
            if Config.use_mixup and np.random.rand() < Config.mixup_prob:
                images, labels = binary_mixup(images, labels, Config.mixup_alpha)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / Config.accum_steps
            
            if not torch.isfinite(loss):
                if is_main_process(rank): print("Warning: Non-finite loss")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            
            if (i + 1) % Config.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if model_ema is not None:
                    model_ema.update(model)
            
            train_loss_sum += loss.item() * Config.accum_steps
            total_batches += 1
            
            if is_main_process(rank):
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=loss.item() * Config.accum_steps, lr=f"{current_lr:.2e}")
        
        scheduler.step()
        
        # Validation
        eval_model = model_ema.module if model_ema is not None else model
        dist.barrier()
        
        epoch_loss, acc, auc, thr, _, _ = validate(eval_model, val_loader, criterion, device)
        
        metrics = torch.tensor([epoch_loss, acc, auc, thr], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world_size
        
        avg_val_loss = metrics[0].item()
        avg_val_acc = metrics[1].item()
        avg_val_auc = metrics[2].item()
        avg_val_thr = metrics[3].item()
        
        if is_main_process(rank):
            print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.4f} | AUC: {avg_val_auc:.4f} (Thr: {avg_val_thr:.2f})")
            
            if avg_val_auc > best_auc_score:
                best_auc_score = avg_val_auc
                final_best_thr = avg_val_thr
                if not os.path.exists(Config.output_dir):
                    os.makedirs(Config.output_dir)
                
                save_model = model_ema.module if model_ema is not None else model.module
                torch.save(save_model.state_dict(), os.path.join(Config.output_dir, "best_model.pth"))
                
                with open(os.path.join(Config.output_dir, "best_threshold.txt"), "w") as f:
                    f.write(str(final_best_thr))
                print(f"  >>> Best AUC Updated! Saved Model. Thr: {final_best_thr:.2f}")

    # Inference
    dist.barrier()
    if is_main_process(rank):
        print(f"\nStarting Inference using 4-way TTA with Threshold: {final_best_thr:.4f}")
        model_best = BreastCancerModel(Config.model_name, pretrained=False).to(device)
        model_best.load_state_dict(torch.load(os.path.join(Config.output_dir, "best_model.pth")))
        model_best.eval()
        
        test_dataset = BreastCancerDataset(Config.test_dir, transform=transforms_dict["valid"], is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)
        
        predictions = []
        with torch.no_grad():
            for images, filenames in tqdm(test_loader, desc="Inference"):
                images = images.to(device)
                
                # 4-way TTA
                out1 = model_best(images)
                out2 = model_best(torch.flip(images, dims=[3])) # H
                out3 = model_best(torch.flip(images, dims=[2])) # V
                out4 = model_best(torch.flip(images, dims=[2, 3])) # HV
                
                probs = (torch.sigmoid(out1) + torch.sigmoid(out2) + torch.sigmoid(out3) + torch.sigmoid(out4)) / 4.0
                
                probs = probs.cpu().numpy().flatten()
                probs = np.nan_to_num(probs, nan=0.0)
                
                preds = (probs >= final_best_thr).astype(int)
                
                for fn, p in zip(filenames, preds):
                    image_id = os.path.splitext(fn)[0]
                    predictions.append([image_id, p])
                    
        df = pd.DataFrame(predictions, columns=["image_id", "target"])
        df = df.sort_values("image_id")
        save_path = os.path.join(Config.output_dir, "submission.csv")
        df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    dist.destroy_process_group()

if __name__ == "__main__":
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1
        
    if world_size > 1:
        port = find_free_port()
        mp.spawn(main_worker, args=(world_size, port), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, find_free_port())