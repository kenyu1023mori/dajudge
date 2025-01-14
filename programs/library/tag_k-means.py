# k-means++（Word2Vec）
from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import emoji

# CSVファイルのパスを指定
file_path = "../../data/data.csv"
# CSVファイルを読み込む
df = pd.read_csv(file_path)

# 'count' 列を整数型に変換（エラーをNaNに変換）
df['count'] = pd.to_numeric(df['count'], errors='coerce')

# NaNを含む行を削除
df = df.dropna(subset=['count'])

# 'count' 列を再度整数型に変換
df['count'] = df['count'].astype(int)

# 'tags' 列の値を ':' で分割し、すべてのタグを抽出してリストに格納
tags_list = df[df['count'] >= 2]['tags'].apply(lambda x: str(x).split(':')).explode().unique()

# 絵文字を除外する関数
def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")  # 絵文字を空文字に置き換え

# 絵文字を除去した単語リストを作成
filtered_tags_list = [remove_emojis(tag) for tag in tags_list if remove_emojis(tag) != '']

# Word2Vecモデルの学習
# Word2Vecは単語間の関係性を学習するため、リストを文書（文章リスト）として扱う必要がある
# tags_list自体を学習用データとするため、単語を直接リスト化する
sentences = [[tag] for tag in filtered_tags_list]

# Word2Vecモデルの設定と学習
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)  # sg=1はSkip-Gramを指定

# 単語をベクトル化
word_vectors = np.array([word2vec_model.wv[word] for word in filtered_tags_list])

# K-means++でクラスタリング
num_clusters = 10000
kmeans = KMeans(n_clusters=num_clusters, init="k-means++", random_state=42)
kmeans.fit(word_vectors)

# クラスタリング結果を取得
clusters = kmeans.labels_

# 結果を表示
clustered_tags = {}
for word, cluster_id in zip(filtered_tags_list, clusters):
    if cluster_id not in clustered_tags:
        clustered_tags[cluster_id] = []
    clustered_tags[cluster_id].append(word)

# クラスタごとの単語を出力
for cluster_id, words in clustered_tags.items():
    print(f"Cluster {cluster_id}: {words}")

# クラスタごとの単語をCSVファイルに保存
clustered_tags_df = pd.DataFrame([(cluster_id, word) for cluster_id, words in clustered_tags.items() for word in words], columns=['ClusterID', 'Tag'])
clustered_tags_df.to_csv("../metrics/clustered_tags.csv", index=False)

# 類似タグをまとめる関数
def merge_similar_tags(clustered_tags):
    merged_tags = {}
    for cluster_id, words in clustered_tags.items():
        representative_tag = words[0]  # クラスタ内の最初の単語を代表タグとする
        for word in words:
            merged_tags[word] = representative_tag
    return merged_tags

# 類似タグをまとめる
merged_tags = merge_similar_tags(clustered_tags)

# 類似タグをまとめた結果をCSVファイルに保存
merged_tags_df = pd.DataFrame(list(merged_tags.items()), columns=['OriginalTag', 'MergedTag'])
merged_tags_df.to_csv("../metrics/merged_tags.csv", index=False)

# 結果を表示
for original_tag, merged_tag in merged_tags.items():
    print(f"Original Tag: {original_tag}, Merged Tag: {merged_tag}")
