import pandas as pd
import os

# ==========================================
# 設定
# ==========================================
folder_path = "./result_refined_multi_gpu"
path_highres_folder = "./result_refined_multi_gpu_768"

# 読み込み
df_eff  = pd.read_csv(os.path.join(folder_path, "submission_eff.csv"))
df_prob = pd.read_csv(os.path.join(path_highres_folder, "submission_highres_prob.csv"))

# ==========================================
# 1. 守備重視: 偽陽性を消す (High-Resの得意技)
# ==========================================
# EfficientNetが「1」でも、High-Resが「0.4以下」なら「0」にする
# (統計では25枚くらい該当します)
sub_fp = df_eff.copy()
count_fp = 0
for i in range(len(sub_fp)):
    if sub_fp.loc[i, "target"] == 1 and df_prob.loc[i, "target"] < 0.40:
        sub_fp.loc[i, "target"] = 0
        count_fp += 1

name_fp = "submission_logic_FIX_FP.csv"
sub_fp.to_csv(os.path.join(folder_path, name_fp), index=False)
print(f"作成: {name_fp} -> {count_fp} 枚を「正常(0)」に修正しました")

# ==========================================
# 2. 攻撃重視: 見逃しを拾う
# ==========================================
# EfficientNetが「0」でも、High-Resが「0.7以上」なら「1」にする
# (統計では16枚くらい該当します)
sub_fn = df_eff.copy()
count_fn = 0
for i in range(len(sub_fn)):
    if sub_fn.loc[i, "target"] == 0 and df_prob.loc[i, "target"] > 0.70:
        sub_fn.loc[i, "target"] = 1
        count_fn += 1

name_fn = "submission_logic_FIX_FN.csv"
sub_fn.to_csv(os.path.join(folder_path, name_fn), index=False)
print(f"作成: {name_fn} -> {count_fn} 枚を「がん(1)」に修正しました")

# ==========================================
# 3. ハイブリッド (両方やる)
# ==========================================
sub_mix = df_eff.copy()
for i in range(len(sub_mix)):
    # FP削除
    if sub_mix.loc[i, "target"] == 1 and df_prob.loc[i, "target"] < 0.40:
        sub_mix.loc[i, "target"] = 0
    # FN救出
    elif sub_mix.loc[i, "target"] == 0 and df_prob.loc[i, "target"] > 0.70:
        sub_mix.loc[i, "target"] = 1

name_mix = "submission_logic_HYBRID.csv"
sub_mix.to_csv(os.path.join(folder_path, name_mix), index=False)
print(f"作成: {name_mix} -> 攻守両方を適用しました")
print("="*50)