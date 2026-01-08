import pandas as pd
import os

# ========================================================
# グランド・フィナーレ: 89%基盤 + 最強の救出
# ========================================================
input_dir = r"D:\puresotu\workespace\nakayama_ken-main\nakayama_ken-main\pure_nakayamaken"

# 1. 89%スコアの基盤となる3モデル
file_ace  = "my_best_78.csv"
file_team = "team_member_77.csv"
file_b4   = "submit_effnet_b4_Mixup_300ep_thr_0.30.csv"

# 2. 今回作った最強の救出用ファイル (EffNet + ConvNeXt)
file_rescue = "final_Soft_Voting_100ep.csv"

try:
    sub_ace    = pd.read_csv(os.path.join(input_dir, file_ace))
    sub_team   = pd.read_csv(os.path.join(input_dir, file_team))
    sub_b4     = pd.read_csv(os.path.join(input_dir, file_b4))
    sub_rescue = pd.read_csv(os.path.join(input_dir, file_rescue))
    print(">> 全ファイル読み込み成功！")
except FileNotFoundError as e:
    print(f"ファイル不足エラー: {e}")
    exit()

target_col = 'target'
# 整形
for df in [sub_ace, sub_team, sub_b4, sub_rescue]:
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False).astype(int)

# --- ロジック ---
# 1. 多数決 (89%の土台)
vote_trio = sub_ace[target_col] + sub_team[target_col] + sub_b4[target_col]

# 2. 救出判定
pred_rescue = sub_rescue[target_col]

final_pred = []

for v, res in zip(vote_trio, pred_rescue):
    # ① 3人中2人以上が賛成なら採用 (89%の強さを維持)
    if v >= 2:
        final_pred.append(1)
        
    # ② 投票では負けたが、高画質タッグ(Eff+Conv)が「黒」と言った場合
    elif res == 1:
        final_pred.append(1)
        
    else:
        final_pred.append(0)

submission = sub_ace.copy()
submission[target_col] = final_pred

output_name = "final_Grand_Finale.csv"
save_path = os.path.join(input_dir, output_name)
submission.to_csv(save_path, index=False)

# 診断
count_final = sum(final_pred)

print("="*50)
print(f"完了: {output_name}")
print(f"判定 '1' の枚数: {count_final} 枚")
print("-" * 30)

if 168 <= count_final <= 180:
    print("★判定: 完璧です！ 89%の168枚に、新たな発見が上乗せされている可能性があります。")
elif count_final == 168:
    print("★判定: 89%の時と同じ枚数です。中身が入れ替わって精度が上がっていることに期待！")
else:
    print(f"★判定: 枚数は {count_final} です。")
print("="*50)