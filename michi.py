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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
import random
import socket

# 警告抑制
warnings.filterwarnings("ignore")

# ==========================================
# 設定 (Configuration)
# ==========================================
class Config:
    seed = 1234
    
    img_size = 640
    num_classes = 1
    
    # モデル設定
    model_name = "convnextv2_base.fcmae_ft_in22k_in1k"
    
    # 学習設定
    batch_size = 2        
    accum_steps = 8       
    
    epochs = 30
    lr = 2e-4
    min_lr = 1e-6
    weight_decay = 1e-4
    max_grad_norm = 1.0 
    
    # 【変更点】MixUp復活 (データが増えたので有効)
    use_mixup = True      
    mixup_p = 0.5         
    mixup_alpha = 0.4     
    cutmix_alpha = 1.0    
    
    # パス
    train_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\train"
    val_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\valid"
    test_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\nakayamaken\BreastCancer\test"
    output_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\result_final_phase6"

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
# MixUp / CutMix Functions
# ==========================================
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

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
# Dataset & Transforms
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
            A.Resize(Config.img_size, Config.img_size),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]),
        "valid": A.Compose([
            A.Resize(Config.img_size, Config.img_size),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5), 
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    }

# ==========================================
# Validation Function (Accuracy探索)
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
            if torch.isnan(probs).any():
                probs = torch.nan_to_num(probs, nan=0.0)
            
            all_probs.append(probs)
            all_labels.append(labels)
            
    # Gather all results
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    # 0除算回避
    dataset_len = len(loader.dataset)
    if dataset_len == 0:
        return 0.0, 0.0, 0.0, 0.5, [], []

    epoch_loss = running_loss / dataset_len
    
    # CPUへ
    y_true = all_labels.cpu().numpy()
    y_prob = all_probs.cpu().numpy()
    
    # AUC
    if len(np.unique(y_true)) > 1:
        val_auc = roc_auc_score(y_true, y_prob)
    else:
        val_auc = 0.0
        
    # Accuracy Maximization (閾値探索)
    best_acc = 0.0
    best_thr = 0.5
    for thr in np.arange(0.01, 1.00, 0.01):
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
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_seed(Config.seed, rank)
    
    # Dataset
    transforms_dict = get_transforms()
    train_dataset = BreastCancerDataset(Config.train_dir, classes=["0", "1"], transform=transforms_dict["train"])
    val_dataset = BreastCancerDataset(Config.val_dir, classes=["0", "1"], transform=transforms_dict["valid"])
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, sampler=val_sampler, num_workers=0, pin_memory=True, drop_last=False)
    
    # 重み計算
    if is_main_process(rank):
        all_labels = [item[1] for item in train_dataset.data]
        num_pos = sum(all_labels)
        num_neg = len(all_labels) - num_pos
        weight_val = num_neg / max(num_pos, 1.0)
        print(f"Data Stats -> Neg: {num_neg}, Pos: {num_pos}, Calculated Pos Weight: {weight_val:.4f}")
    else:
        # Rank 0以外も定義だけ必要（値は適当で良いが同期させるのがベスト）
        weight_val = 1.0 
    
    # ブロードキャストで重みを統一
    weight_tensor = torch.tensor([weight_val], device=device)
    dist.broadcast(weight_tensor, src=0)
    pos_weight = weight_tensor
    
    # Model
    model = BreastCancerModel(Config.model_name).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.lr,
        epochs=Config.epochs,
        steps_per_epoch=len(train_loader) // Config.accum_steps,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # 安定したBCEに戻す
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.cuda.amp.GradScaler()
    
    best_acc_score = 0.0
    final_best_thr = 0.5
    
    if is_main_process(rank):
        print(f"Start Training: {Config.model_name} on {world_size} GPUs (Acc Optimization)")
    
    # --- Training Loop ---
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
            
            # --- MixUp / CutMix Logic ---
            do_mix = False
            if Config.use_mixup and np.random.random() < Config.mixup_p:
                do_mix = True
                if np.random.random() < 0.5:
                    images, y_a, y_b, lam = mixup_data(images, labels, Config.mixup_alpha)
                else:
                    images, y_a, y_b, lam = cutmix_data(images, labels, Config.cutmix_alpha)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                
                if do_mix:
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                else:
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
                scheduler.step()
            
            train_loss_sum += loss.item() * Config.accum_steps
            total_batches += 1
            
            if is_main_process(rank):
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=loss.item() * Config.accum_steps, lr=f"{current_lr:.2e}")
        
        dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
        if total_batches > 0:
            avg_train_loss = train_loss_sum.item() / (total_batches * world_size)
        else:
            avg_train_loss = 0.0
        
        # Validation (各Rankで計算し、Rank 0 で集計して表示する簡易実装は関数内にあり)
        # 正確なThreshold探索のため、Rank 0 にデータを集めて計算する実装にする
        model.eval()
        
        # Rankごとに推論結果を収集
        local_preds = []
        local_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).squeeze()
                if torch.isnan(probs).any(): probs = torch.nan_to_num(probs, nan=0.0)
                local_preds.append(probs)
                local_labels.append(labels.to(device))
                
        if len(local_preds) > 0:
            local_preds = torch.cat(local_preds)
            local_labels = torch.cat(local_labels)
        else:
            local_preds = torch.tensor([]).to(device)
            local_labels = torch.tensor([]).to(device)

        # 全Rankの結果を集約
        gathered_preds = [torch.zeros_like(local_preds) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(local_labels) for _ in range(world_size)]
        dist.all_gather(gathered_preds, local_preds)
        dist.all_gather(gathered_labels, local_labels)
        
        if is_main_process(rank):
            all_preds = torch.cat(gathered_preds).cpu().numpy()
            all_labels = torch.cat(gathered_labels).cpu().numpy()
            
            if np.isnan(all_preds).any(): all_preds = np.nan_to_num(all_preds, nan=0.5)
            
            # 閾値探索
            best_acc = 0.0
            best_thr = 0.5
            # 全データで計算
            y_true = all_labels
            y_prob = all_preds
            
            if len(np.unique(y_true)) > 1:
                val_auc = roc_auc_score(y_true, y_prob)
            else:
                val_auc = 0.0

            # 0.01刻みでBest Accを探す
            for thr in np.arange(0.01, 1.00, 0.01):
                y_pred_tmp = (y_prob >= thr).astype(int)
                acc_tmp = accuracy_score(y_true, y_pred_tmp)
                if acc_tmp > best_acc:
                    best_acc = acc_tmp
                    best_thr = thr
            
            print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val Acc: {best_acc:.4f} (Thr: {best_thr:.2f}) | AUC: {val_auc:.4f}")
            
            # Accuracyが改善したら保存
            if best_acc > best_acc_score:
                best_acc_score = best_acc
                final_best_thr = best_thr
                
                if not os.path.exists(Config.output_dir):
                    os.makedirs(Config.output_dir)
                torch.save(model.module.state_dict(), os.path.join(Config.output_dir, "best_model.pth"))
                # 閾値を保存
                with open(os.path.join(Config.output_dir, "best_threshold.txt"), "w") as f:
                    f.write(str(final_best_thr))
                print(f"  >>> Best Accuracy Updated! Saved Model & Thr: {final_best_thr:.2f}")

    # --- Inference with TTA ---
    dist.barrier()
    if is_main_process(rank):
        print(f"\nStarting Inference on Rank 0 using Threshold: {final_best_thr:.4f}")
        model_best = BreastCancerModel(Config.model_name, pretrained=False).to(device)
        model_best.load_state_dict(torch.load(os.path.join(Config.output_dir, "best_model.pth")))
        model_best.eval()
        
        test_dataset = BreastCancerDataset(Config.test_dir, transform=transforms_dict["valid"], is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=Config.batch_size*2, shuffle=False, num_workers=0)
        
        predictions = []
        with torch.no_grad():
            for images, filenames in tqdm(test_loader, desc="Inference"):
                images = images.to(device)
                
                # TTA
                out1 = model_best(images)
                out2 = model_best(torch.flip(images, dims=[3]))
                
                prob1 = torch.sigmoid(out1).cpu().numpy().flatten()
                prob2 = torch.sigmoid(out2).cpu().numpy().flatten()
                
                prob1 = np.nan_to_num(prob1, nan=0.0)
                prob2 = np.nan_to_num(prob2, nan=0.0)
                
                final_probs = (prob1 + prob2) / 2.0
                
                # 最適化された閾値で予測
                preds = (final_probs >= final_best_thr).astype(int)
                
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
    
    print(f"Using {world_size} GPUs")
    if world_size > 1:
        port = find_free_port()
        mp.spawn(main_worker, args=(world_size, port), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, find_free_port())