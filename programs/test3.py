import pandas as pd

# CSVファイルのパス
file_path = "../../data/test.csv"

# データの読み込み
data = pd.read_csv(file_path, header=None)

# シャッフル
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 同じファイルに上書き保存
data.to_csv(file_path, header=False, index=False, encoding="utf-8-sig")

print(f"ごちゃ混ぜにしたデータを {file_path} に保存しました！")
