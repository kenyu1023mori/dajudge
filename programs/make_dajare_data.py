import pandas as pd
import random

# 元のCSVファイルの読み込み
data = pd.read_csv('../../data/data.csv')

# count列を数値型に変換（エラーが出た場合はNaNに置き換える）
data['count'] = pd.to_numeric(data['count'], errors='coerce')

# 条件: countが2以上のものを抽出
filtered_data = data[data['count'] >= 2]

# ランダムにデータを混ぜる
shuffled_data = filtered_data.sample(frac=1, random_state=random.randint(1, 10000))

# 必要な列(dajare, score)のみ抽出
result = shuffled_data[['dajare', 'score']]

# 新しいCSVファイルとして保存
result.to_csv('filtered_dajare.csv', index=False, encoding='utf-8')

print("新しいCSVファイルが作成されました: filtered_dajare.csv")
