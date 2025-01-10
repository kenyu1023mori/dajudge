import pandas as pd

# data.csvから、各スコア範囲ごとにランダムに指定された個数ずつ選択して、新しいCSVファイルを作成する

# CSVファイルの読み込み
input_file_path = "/home/group4/data/filtered_by_shareka.csv"
output_file_path = "/home/group4/data/evenly_after_shareka.csv"

# データの読み込み
data = pd.read_csv(input_file_path)

# スコア列を数値型に変換
data['score'] = pd.to_numeric(data['score'], errors='coerce')

# スコア範囲ごとにランダムに指定された個数選択
score_ranges = [(1.0, 1.9, 2000), (2.0, 2.9, 2000), (3.0, 3.9, 3000), (4.0, 4.9, 3000)]
sampled_data = []

for score_min, score_max, sample_size in score_ranges:
    filtered_data = data[(data['score'] >= score_min) & (data['score'] <= score_max)]
    sampled_data.append(filtered_data.sample(n=sample_size, random_state=42))

# 全部まとめて一つのデータフレームにする
combined_data = pd.concat(sampled_data)

# ランダムに混ぜる
shuffled_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# 新しいCSVファイルに書き出す
shuffled_data.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"新しいCSVファイルが作成されました: {output_file_path}")
