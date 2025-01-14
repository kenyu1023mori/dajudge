import pandas as pd

# CSVファイルの読み込み
file_path = "/home/group4/data/filtered_by_shareka.csv"
data = pd.read_csv(file_path)

# スコア範囲ごとのデータ数をカウント
score_ranges = [(1.0, 1.9), (2.0, 2.9), (3.0, 3.9), (4.0, 4.9)]
counts = {}

for score_min, score_max in score_ranges:
    count = data[(data['score'] >= score_min) & (data['score'] <= score_max)].shape[0]
    counts[f"{score_min}-{score_max}"] = count

# 結果を表示
for score_range, count in counts.items():
    print(f"Score {score_range}: {count} items")
