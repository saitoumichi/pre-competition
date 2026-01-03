import pandas as pd
import os

# ==========================================
# 設定
# ==========================================
folder_path = "./result_refined_multi_gpu"

file_base  = "submission_base.csv"       # Base
file_small = "submission_small.csv"      # Small
file_eff   = "submission_eff.csv"        # Eff (75%)

# ==========================================
# 読み込み
# ==========================================
path_base  = os.path.join(folder_path, file_base)
path_small = os.path.join(folder_path, file_small)
path_eff   = os.path.join(folder_path, file_eff)

df_base  = pd.read_csv(path_base)
df_small = pd.read_csv(path_small)
df_eff   = pd.read_csv(path_eff)

# ==========================================
# 確率計算 (70:20:10)
# ==========================================
ensemble_prob = (
    df_eff["target"]   * 0.70 + 
    df_base["target"]  * 0.20 + 
    df_small["target"] * 0.10
)

# ==========================================
# ★ 0.7の壁を超える閾値設定
# ==========================================
# 0.75: Effが1でも、Base/Smallが強く否定すれば「0」になるライン
# 0.80: Effが1かつ、Base/Smallもそこそこ賛成しないと「1」にならないライン
thresholds = [0.75, 0.80]

print("="*50)
for thr in thresholds:
    df_submit = df_eff.copy()
    
    # 確率が閾値以上なら1
    df_submit["target"] = (ensemble_prob >= thr).astype(int)
    
    # ファイル名
    filename = f"submission_WEIGHTED_thr_{thr}.csv"
    save_path = os.path.join(folder_path, filename)
    df_submit.to_csv(save_path, index=False)
    
    # 0.5（今の75%）との違いを確認
    # これで「0枚」以外が出れば勝利の可能性あり！
    diff_count = (df_submit["target"] != (ensemble_prob >= 0.5).astype(int)).sum()
    
    print(f"作成完了: {filename}")
    print(f" -> 閾値0.5との違い: {diff_count} 枚の判定が「正常(0)」に変わりました")

print("="*50)