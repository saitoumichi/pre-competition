import pandas as pd

# ========================================================
# ★これが「82%」を出した最強の3人組です
# ========================================================
# 1. エース（ConvNeXt）
file_ace = 'my_best_78.csv'
# 2. チームメイト（EfficientNet）
file_team = 'team_member_77.csv'
# 3. ★MVP★ 通常の30エポック版（これが一番相性が良かった！）
# ※Mixup版(0.75)ではなく、こっち(0.60)を使います
file_mvp = 'submit_effnet_b4_TTA_thr_0.60.csv'
# ========================================================

print("ファイルを読み込んでいます...")
try:
    sub1 = pd.read_csv(file_ace)
    sub2 = pd.read_csv(file_team)
    sub3 = pd.read_csv(file_mvp)
except FileNotFoundError:
    print("★エラー：ファイルが見つかりません。")
    print("「submit_effnet_b4_TTA_thr_0.60.csv」があるか確認してください！")
    exit()

target_col = 'target' 

# お掃除
for sub in [sub1, sub2, sub3]:
    sub[target_col] = sub[target_col].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False).astype(int)

# 相関チェック（0.42前後なら成功）
print(f"相関係数(エース vs MVP): {sub1[target_col].corr(sub3[target_col]):.4f}")

# シンプルな多数決（これが一番強かった！）
total_vote = sub1[target_col] + sub2[target_col] + sub3[target_col]
submission = sub1.copy()
submission[target_col] = (total_vote >= 2).astype(int)

# 誇り高きファイル名で保存
output_name = 'submission_BEST_SCORE_82.csv'
submission.to_csv(output_name, index=False)

print(f"--------------------------------------------------")
print(f"最強ファイル復活完了！: {output_name}")
print(f"いろいろ試しましたが、結局これがNo.1でした。")
print(f"これを最終提出として送り出しましょう！")
print(f"--------------------------------------------------")