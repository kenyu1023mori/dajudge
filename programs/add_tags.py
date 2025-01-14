import pandas as pd
import numpy as np
import sqlite3
import torch
from transformers import BertJapaneseTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# SQLiteデータベースからタグを読み込む
db_path = "/home/group4/data/final/tags.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("SELECT * FROM tags")
tags = c.fetchall()
conn.close()

# タグのリストを作成
tag_list = [tag[1] for tag in tags]

# BERTモデルとトークナイザーの準備
bert_model_name = "cl-tohoku/bert-base-japanese-v3"
tokenizer = BertJapaneseTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# タグのBERT埋め込み生成関数
def get_bert_embeddings(texts, tokenizer, model, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# タグの埋め込み生成
tag_embeddings = get_bert_embeddings(tag_list, tokenizer, bert_model)

# CSVファイルの読み込み
csv_file_path = "/home/group4/data/final/dajare_dataset_without_tags.csv"
data = pd.read_csv(csv_file_path)

# dajare列の文章の埋め込み生成
dajare_sentences = data['dajare'].astype(str).tolist()

# コサイン類似度を計算し、最も類似度の高い3つのタグを追加
def get_top_n_similar_tags(dajare_embedding, tag_embeddings, tag_list, n=3):
    similarities = cosine_similarity([dajare_embedding], tag_embeddings)[0]
    top_n_indices = similarities.argsort()[-n:][::-1]
    return [tag_list[i] for i in top_n_indices]

data['tag1'] = ""
data['tag2'] = ""
data['tag3'] = ""

batch_size = 100  # バッチサイズを設定

for i in range(0, len(dajare_sentences), batch_size):
    batch_sentences = dajare_sentences[i:i + batch_size]
    batch_embeddings = get_bert_embeddings(batch_sentences, tokenizer, bert_model)
    for j, dajare_embedding in enumerate(batch_embeddings):
        top_tags = get_top_n_similar_tags(dajare_embedding, tag_embeddings, tag_list)
        data.at[i + j, 'tag1'] = top_tags[0]
        data.at[i + j, 'tag2'] = top_tags[1]
        data.at[i + j, 'tag3'] = top_tags[2]

# 新しいCSVファイルとして保存
output_file_path = "/home/group4/data/dajare_dataset.csv"
data.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"新しいCSVファイルが作成されました: {output_file_path}")
