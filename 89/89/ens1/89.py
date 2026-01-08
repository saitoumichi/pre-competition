import pandas as pd
import os

# ========================================================
# 復元: final_HighRes_Rescue.csv (Best 89%)
# ========================================================
input_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\pure_nakayamaken"
# 高画質モデル(100ep)の場所
eff_dir   = r"./result_effnetv2_384_100ep"

# 1. 88%のエースチーム (Ace + Team + B4)
file_ace  = "my_best_78.csv"
file_team = "team_member_77.csv"
file_b4   = "submit_effnet_b4_Mixup_300ep_thr_0.30.csv"

# 2. 今回の最強モデル (EffNetV2 384px)
file_eff  = "submission_effnetv2_384.csv"

try:
    sub_ace  = pd.read_csv(os.path.join(input_dir, file_ace))
    sub_team = pd.read_csv(os.path.join(input_dir, file_team))
    sub_b4   = pd.read_csv(os.path.join(input_dir, file_b4))
    sub_eff  = pd.read_csv(os.path.join(eff_dir, file_eff))
    print(">> データ読み込み完了！")
except FileNotFoundError as e:
    print(f"ファイルが見つかりません: {e}")
    exit()

target_col = 'target'
for df in [sub_ace, sub_team, sub_b4, sub_eff]:
    df[target_col] = df[target_col].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False).astype(int)

# --- ロジック ---

# 1. 88%チームの意見 (2票以上で採用)
vote_88 = sub_ace[target_col] + sub_team[target_col] + sub_b4[target_col]
pred_88 = (vote_88 >= 2).astype(int)

# 2. 高画質救出作戦 (EffNetV2が「1」なら無条件で救い上げる)
final_pred = []
rescue_count = 0

for p88, p_eff in zip(pred_88, sub_eff[target_col]):
    if p88 == 1:
        # 元々88%チームが見つけていた
        final_pred.append(1)
    elif p_eff == 1:
        # 88%チームは見逃したが、高画質モデルが発見した（救出！）
        final_pred.append(1)
        rescue_count += 1
    else:
        final_pred.append(0)

submission = sub_ace.copy()
submission[target_col] = final_pred

output_name = "final_HighRes_Rescue.csv"
save_path = os.path.join(input_dir, output_name)
submission.to_csv(save_path, index=False)

# 確認
count_final = sum(final_pred)
print("="*50)
print(f"作成完了: {output_name}")
print(f"最終枚数: {count_final} 枚 (目標: 168枚)")
print("="*50)