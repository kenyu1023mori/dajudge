import pandas as pd
import random

# 元のCSVファイルの読み込み
data = pd.read_csv('../../data/filtered_dajare.csv')

# スコアの分布を調べる
def score_distribution(data, bins):
    data['score'] = pd.to_numeric(data['score'], errors='coerce')  # スコアを数値型に変換
    distribution = pd.cut(data['score'], bins=bins).value_counts().sort_index()
    for interval, count in distribution.items():
        print(f"{interval.left}以上{interval.right}未満が{count}個")

# スコアの分布を出力する区間
bins = [1, 2, 3, 4, 5]  # 区間の定義
score_distribution(data, bins)
