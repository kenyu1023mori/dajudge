import numpy as np
import torch
import torch.nn as nn
import os
import MeCab
import pykakasi
import fasttext
from transformers import BertJapaneseTokenizer, BertModel

# 必要な変数とパスを設定
version = "v2.04"
load_dir = f"../models/{version}"
fasttext_model_path = "../models/cc.ja.300.bin"
bert_model_name = "cl-tohoku/bert-base-japanese"

# MeCabの設定
mecab = MeCab.Tagger("-Owakati")  # 単語を分かち書き形式で取得
kakasi = pykakasi.kakasi()  # 音韻解析用

# ニューラルネットワークモデルのクラス定義
class DajarePredictor(nn.Module):
    def __init__(self):
        super(DajarePredictor, self).__init__()
        input_size = 768 + 3 + 300  # BERT + 音韻特徴量 + fastText
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# BERTモデルとトークナイザーのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# fastTextモデルをロード
fasttext_model = fasttext.load_model(fasttext_model_path)

# 音韻特徴量を生成
def phonetic_features(sentence):
    result = kakasi.convert(sentence)
    romaji = " ".join([item["hepburn"] for item in result])
    length = len(romaji.split())  # 音節数
    vowels = sum(1 for char in romaji if char in "aeiou")  # 母音の数
    consonants = len(romaji.replace(" ", "")) - vowels  # 子音の数
    return [length, vowels, consonants]

# 文をBERT埋め込みに変換
def get_bert_embeddings(sentences, tokenizer, model):
    inputs = tokenizer(sentences, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# fastText埋め込みを取得
def get_fasttext_embeddings(sentence, model):
    words = mecab.parse(sentence).strip().split()
    word_embeddings = [model.get_word_vector(word) for word in words]
    if word_embeddings:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(300)

# モデルのロード
model = DajarePredictor()
model_path = os.path.join(load_dir, "Dajudge_fold_1.pth")
model.load_state_dict(torch.load(model_path))
model.eval()

# 入力したダジャレに対してモデルのスコアを出力する関数
def predict_score(input_text, model, tokenizer, bert_model, fasttext_model, mecab, kakasi):
    bert_embedding = get_bert_embeddings([input_text], tokenizer, bert_model)
    phonetic_feature = np.array([phonetic_features(input_text)])
    fasttext_embedding = np.array([get_fasttext_embeddings(input_text, fasttext_model)])
    input_vector = np.hstack((bert_embedding, phonetic_feature, fasttext_embedding))
    input_tensor = torch.tensor(input_vector, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).squeeze().item()
        print(f"Predicted Score: {prediction}")

# ユーザー入力処理
while True:
    input_text = input("Enter a Dajare (or type 'q' to quit): ")
    if input_text.lower() == 'q':
        break
    predict_score(input_text, model, tokenizer, bert_model, fasttext_model, mecab, kakasi)
