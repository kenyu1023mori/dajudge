import numpy as np
import torch
import torch.nn as nn
import os
from transformers import BertJapaneseTokenizer, BertModel
import MeCab
import fasttext
import pykakasi

# 必要な変数とパスを設定
version = "v1.23"
load_dir = f"../models/{version}"
bert_model_name = "cl-tohoku/bert-base-japanese"
fasttext_model_path = "../models/cc.ja.300.bin"

# BERTモデルとトークナイザーの初期化
tokenizer = BertJapaneseTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# fastTextモデルのロード
fasttext_model = fasttext.load_model(fasttext_model_path)

# MeCabの設定
mecab = MeCab.Tagger("-Owakati")  # 単語を分かち書き形式で取得

# pykakasiの初期化
kakasi = pykakasi.kakasi()

# ニューラルネットワークモデルのクラス定義
class DajarePredictor(nn.Module):
    def __init__(self):
        super(DajarePredictor, self).__init__()
        self.fc1 = nn.Linear(768 + 3 + 300, 256)  # BERT + 音韻特徴量 + fastText
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 文をBERTの埋め込みに変換する関数
def get_bert_embedding(sentence, tokenizer, model):
    inputs = tokenizer([sentence], return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# fastText埋め込みを取得する関数
def get_fasttext_embedding(sentence, model):
    words = mecab.parse(sentence).strip().split()
    word_embeddings = [model.get_word_vector(word) for word in words]
    if word_embeddings:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(300)

# 音韻特徴量を生成する関数
def phonetic_features(sentence):
    result = kakasi.convert(sentence)
    romaji = " ".join([item["hepburn"] for item in result])
    length = len(romaji.split())  # 音節数
    vowels = sum(1 for char in romaji if char in "aeiou")  # 母音の数
    consonants = len(romaji.replace(" ", "")) - vowels  # 子音の数
    return [length, vowels, consonants]

# モデルのロード
models = [DajarePredictor() for _ in range(5)]
for fold in range(5):
    model_path = os.path.join(load_dir, f"Dajudge_fold_{fold+1}.pth")
    models[fold].load_state_dict(torch.load(model_path, weights_only=True))
    models[fold].eval()

# 入力したダジャレに対してモデルのスコアを出力する関数
def predict_score(input_text, models, tokenizer, bert_model, fasttext_model):
    bert_embedding = get_bert_embedding(input_text, tokenizer, bert_model)
    fasttext_embedding = get_fasttext_embedding(input_text, fasttext_model)
    phonetic_feature = phonetic_features(input_text)
    input_vector = torch.tensor([np.hstack((bert_embedding, phonetic_feature, fasttext_embedding))], dtype=torch.float32)

    with torch.no_grad():
        predictions = [model(input_vector).squeeze() for model in models]
        average_prediction = torch.stack(predictions).mean().item()
        predicted_class = max(1, min(average_prediction, 100))  # 100点満点に対応

        print(f"Predicted Score: {predicted_class}")

# ユーザー入力処理
while True:
    input_text = input("Enter a Dajare (or type 'q' to quit): ")
    if input_text.lower() == 'q':
        break
    predict_score(input_text, models, tokenizer, bert_model, fasttext_model)
