import pandas as pd

# ========================================================
# 1. ファイル名を設定
# ========================================================
file_ace  = 'my_best_78.csv'          # エース (78%)
file_team = 'team_member_77.csv'      # チーム (77%)

# ★今回生まれた「最強のB4 (76.44%)」
# ※パスが長いので、間違えないようにそのまま貼っておきました
file_new_b4 = 'submit_effnet_b4_Mixup_300ep_thr_0.30.csv'
# ========================================================

print("ファイルを読み込んでいます...")
try:
    sub1 = pd.read_csv(file_ace)
    sub2 = pd.read_csv(file_team)
    sub3 = pd.read_csv(file_new_b4)
except FileNotFoundError:
    print("★エラー：ファイルが見つかりません。パスを確認してください！")
    exit()

target_col = 'target' 

# お掃除（念のため）
for sub in [sub1, sub2, sub3]:
    sub[target_col] = sub[target_col].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False).astype(int)

# 相関チェック（ここが重要！）
# 強くなった分、エースと似てしまっていないか確認します
corr_1_3 = sub1[target_col].corr(sub3[target_col])
print(f"相関係数(エース vs 新B4): {corr_1_3:.4f}")

if corr_1_3 < 0.7:
    print("★判定: 相関係数が低い！これは爆伸びの予感です！")
else:
    print("★判定: 相関係数が高いですが、基礎能力が高いので期待できます！")

# アンサンブル（多数決）
total_vote = sub1[target_col] + sub2[target_col] + sub3[target_col]
submission = sub1.copy()
submission[target_col] = (total_vote >= 2).astype(int)

output_name = 'final_ensemble_Super_B4_300ep.csv'
submission.to_csv(output_name, index=False)

print(f"--------------------------------------------------")
print(f"最強アンサンブル完了！ {output_name} ができました。")
print(f"これが今回の自信作です。提出してみましょう！")
print(f"--------------------------------------------------")