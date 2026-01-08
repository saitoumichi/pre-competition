import os
import sys
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score
import warnings

# --- 安定動作設定 ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
warnings.filterwarnings("ignore")

class Config:
    # 作成した5つのモデルファイル情報
    models_config = [
        {'name': 'tf_efficientnetv2_b0', 'size': 224, 'path': 'tf_efficientnetv2_b0_finetuned.pth'},
        {'name': 'tf_efficientnetv2_b1', 'size': 240, 'path': 'tf_efficientnetv2_b1_finetuned.pth'},
        {'name': 'tf_efficientnetv2_b2', 'size': 260, 'path': 'tf_efficientnetv2_b2_finetuned.pth'},
        {'name': 'tf_efficientnetv2_b3', 'size': 300, 'path': 'tf_efficientnetv2_b3_finetuned.pth'},
        {'name': 'tf_efficientnetv2_s',  'size': 380, 'path': 'tf_efficientnetv2_s_finetuned.pth'},
    ]
    
    # パス設定
    base_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main"
    models_dir = os.path.join(base_dir, "models_finetuned")
    
    train_dir = os.path.join(base_dir, r"nakayamaken\BreastCancer\train")
    val_dir = os.path.join(base_dir, r"nakayamaken\BreastCancer\valid")
    test_dir = os.path.join(base_dir, r"nakayamaken\BreastCancer\test")
    output_dir = os.path.join(base_dir, r"result_final_ensemble")
    os.makedirs(output_dir, exist_ok=True)
    
    # アンサンブル学習設定
    batch_size = 8  # 安全のため小さめ
    epochs = 10     # 既に賢いモデルたちなので短くてOK
    lr = 0.001
    base_img_size = 384

# ---------------------------------------------------------
# アンサンブルモデル定義
# ---------------------------------------------------------
class FinetunedEnsemble(nn.Module):
    def __init__(self, num_classes=1):
        super(FinetunedEnsemble, self).__init__()
        
        self.models = nn.ModuleList()
        feature_dims = []
        self.size_keys = []
        
        print("\n>>> Loading Fine-tuned Backbones...")
        
        for cfg in Config.models_config:
            # 1. 骨組みを作る
            model = timm.create_model(cfg['name'], pretrained=False, num_classes=1)
            
            # 2. 特訓した重みをロード
            weight_path = os.path.join(Config.models_dir, cfg['path'])
            try:
                # DataParallelや通常保存など、キー名の違いを吸収してロード
                state_dict = torch.load(weight_path)
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace("module.", "") # module. があれば削除
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                print(f"  [OK] Loaded: {cfg['name']}")
            except Exception as e:
                print(f"  [Error] Failed to load {cfg['name']}: {e}")
                print("  -> (注意) 重みが見つからないため、初期値で代用します")
                model = timm.create_model(cfg['name'], pretrained=True, num_classes=1)

            # 3. 分類層を削除して特徴抽出モードへ
            model.reset_classifier(0)
            
            # 4. 固定 (Freeze)
            for param in model.parameters():
                param.requires_grad = False
                
            self.models.append(model)
            feature_dims.append(model.num_features)
            self.size_keys.append(str(cfg['size']))

        # 結合後の次元数
        total_features = sum(feature_dims)
        
        # 最終分類層 (論文 Figure 5)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(total_features),
            nn.Dropout(0.2),
            nn.Linear(total_features, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x_dict):
        features = []
        
        for i, model in enumerate(self.models):
            key = self.size_keys[i]
            x = x_dict[key] # 辞書から対応サイズの画像を取得
            f = model(x)
            features.append(f)
        
        concat_features = torch.cat(features, dim=1)
        output = self.classifier(concat_features)
        return output

# ---------------------------------------------------------
# データセット (マルチサイズ対応・CPUリサイズ)
# ---------------------------------------------------------
class MultiSizeDataset(Dataset):
    def __init__(self, root_dir, classes=None, is_test=False, augment=False):
        self.root_dir = root_dir
        self.is_test = is_test
        self.augment = augment
        self.data = []
        
        # Augmentation
        if augment:
            self.geom_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
            ])
        else:
            self.geom_transform = None
            
        self.normalize = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        if not is_test:
            self.classes = classes
            if not os.path.exists(root_dir): return
            for class_label in self.classes:
                class_path = os.path.join(self.root_dir, class_label)
                if not os.path.isdir(class_path): continue
                label_index = self.classes.index(class_label)
                files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for f in files:
                    self.data.append((os.path.join(class_path, f), label_index))
        else:
            if os.path.exists(root_dir):
                self.files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            else:
                self.files = []

    def __len__(self):
        return len(self.files) if self.is_test else len(self.data)

    def __getitem__(self, idx):
        if self.is_test:
            fname = self.files[idx]
            path = os.path.join(self.root_dir, fname)
            target = fname
        else:
            path, target = self.data[idx]
            
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((Config.base_img_size, Config.base_img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        if self.geom_transform:
            img = self.geom_transform(image=img)['image']
            
        # 5つのサイズを作成して辞書に格納 (これでCUDAエラーを回避)
        img_dict = {}
        for cfg in Config.models_config:
            size = cfg['size']
            resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
            transformed = self.normalize(image=resized)['image']
            img_dict[str(size)] = transformed
            
        if self.is_test:
            return img_dict, target
        else:
            return img_dict, torch.tensor(target, dtype=torch.float32)

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
def main():
    # GPU 0番のみ使用 (安定性重視)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. データセット
    print("\n--- Preparing Datasets ---")
    train_ds = MultiSizeDataset(Config.train_dir, classes=["0", "1"], augment=True, is_test=False)
    val_ds = MultiSizeDataset(Config.val_dir, classes=["0", "1"], augment=False, is_test=False)
    
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 2. モデル作成 (Step 1の成果をロード)
    model = FinetunedEnsemble(num_classes=1).to(device)
    
    # 3. 学習設定
    optimizer = optim.Adam(model.classifier.parameters(), lr=Config.lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # 4. 学習ループ
    print(f"\n--- Start Final Ensemble Training ({Config.epochs} epochs) ---")
    best_acc = 0.0
    save_path = os.path.join(Config.output_dir, "final_ensemble_model.pth")
    
    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0
        
        for imgs_dict, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}"):
            for k in imgs_dict:
                imgs_dict[k] = imgs_dict[k].to(device)
            lbls = lbls.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            out = model(imgs_dict)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 検証
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for imgs_dict, lbls in val_loader:
                for k in imgs_dict:
                    imgs_dict[k] = imgs_dict[k].to(device)
                out = model(imgs_dict)
                prob = torch.sigmoid(out).view(-1).cpu().numpy()
                preds.extend(prob)
                targets.extend(lbls.numpy())
        
        acc = accuracy_score(targets, (np.array(preds) >= 0.5).astype(int))
        print(f"  Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print("  >>> Best Model Saved!")
            
        scheduler.step(acc)
        
    print(f"\nTraining Finished. Best Val Acc: {best_acc:.4f}")
    
    # 5. 推論フェーズ
    print("\n--- Inference Phase ---")
    
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    # 最適閾値の探索
    print("Searching best threshold...")
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for imgs_dict, lbls in val_loader:
            for k in imgs_dict:
                imgs_dict[k] = imgs_dict[k].to(device)
            out = model(imgs_dict)
            val_preds.extend(torch.sigmoid(out).view(-1).cpu().numpy())
            val_targets.extend(lbls.numpy())
            
    val_preds = np.array(val_preds)
    best_thr = 0.5
    best_score = 0.0
    
    for thr in np.arange(0.01, 1.00, 0.01):
        score = accuracy_score(val_targets, (val_preds >= thr).astype(int))
        if score > best_score:
            best_score = score
            best_thr = thr
            
    print(f"Best Threshold: {best_thr:.2f} (Acc: {best_score:.4f})")
    
    # テストデータ推論
    test_ds = MultiSizeDataset(Config.test_dir, augment=False, is_test=True)
    if len(test_ds) == 0: return
    test_loader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False, num_workers=0)
    
    final_preds = []
    print("Running Inference...")
    with torch.no_grad():
        for imgs_dict, fnames in tqdm(test_loader, desc="Inference"):
            for k in imgs_dict:
                imgs_dict[k] = imgs_dict[k].to(device)
            out = model(imgs_dict)
            prob = torch.sigmoid(out).view(-1).cpu().numpy()
            pred_lbls = (prob >= best_thr).astype(int)
            
            for fn, p in zip(fnames, pred_lbls):
                final_preds.append([os.path.splitext(fn)[0], p])
                
    df = pd.DataFrame(final_preds, columns=["image_id", "target"])
    df = df.sort_values("image_id")
    save_csv = os.path.join(Config.output_dir, f"submission_finetuned_ensemble_thr_{best_thr:.2f}.csv")
    df.to_csv(save_csv, index=False)
    print(f"Done! Saved: {save_csv}")

if __name__ == "__main__":
    main()