import numpy as np
import torch
import torch.nn as nn
import os
import MeCab
import pykakasi
import fasttext
from transformers import BertJapaneseTokenizer, BertModel

# 必要な変数とパスを設定
version = "v3.06"
load_dir = f"../models/{version}"
fasttext_model_path = "../models/cc.ja.300.bin"
bert_model_name = "cl-tohoku/bert-base-japanese-v3"

# MeCabの設定
mecab = MeCab.Tagger("-Owakati")  # 単語を分かち書き形式で取得
kakasi = pykakasi.kakasi()  # 音韻解析用

# ニューラルネットワークモデルのクラス定義
class DajarePredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate):
        super(DajarePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x

# BERTモデルとトークナイザーのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# fastTextモデルをロード
fasttext_model = fasttext.load_model(fasttext_model_path)

# 音韻特徴量を生成
def extract_phonetic_features(yomi):
    romaji = yomi
    vowels = sum(1 for char in romaji if char in "aeiou")
    consonants = len(romaji.replace(" ", "")) - vowels
    length = len(romaji.split())  # 音節数
    repeat_ratio = sum(romaji.count(char) > 1 for char in set(romaji)) / len(set(romaji))
    
    return [length, vowels, consonants, vowels / (consonants + 1e-5), repeat_ratio]

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
input_size = 1073  # Update input size to match dajudge_train.py
# Best parameters: {'hidden_size1': 240, 'hidden_size2': 165, 'hidden_size3': 92, 'hidden_size4': 20, 'dropout_rate': 0.47319861745762953, 'learning_rate': 1.4899714037058526e-05, 'batch_size': 52, 'epochs': 43, 'split_seed': 2122}
hidden_sizes = [240, 165, 92, 20]  # 手動で設定
dropout_rate = 0.47319861745762953  # 手動で設定

model = DajarePredictor(input_size=input_size, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)
model_path = os.path.join(load_dir, "Dajare_best.pth")  # Update model path to match dajudge_train.py
model.load_state_dict(torch.load(model_path, weights_only=True))  # Set weights_only=True to avoid the warning
model.eval()

# ひらがなと記号のみを抽出する関数
def extract_hiragana_and_symbols(text):
    return ''.join([char for char in text if char in "ぁ-んー、。！？"])

# 入力したダジャレに対してモデルのスコアを出力する関数
def predict_score(input_text, model, tokenizer, bert_model, fasttext_model, mecab, kakasi):
    bert_embedding = get_bert_embeddings([input_text], tokenizer, bert_model)
    yomi = mecab.parse(input_text).strip()
    yomi = extract_hiragana_and_symbols(yomi)
    phonetic_feature = np.array([extract_phonetic_features(yomi)])
    fasttext_embedding = np.array([get_fasttext_embeddings(input_text, fasttext_model)])
    input_vector = np.hstack((bert_embedding, fasttext_embedding, phonetic_feature))
    input_tensor = torch.tensor(input_vector, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).squeeze().item()
        print(f"Predicted Score: {prediction * 4 + 1}")  # Adjust score scaling

# ユーザー入力処理
while True:
    input_text = input("Enter a Dajare (or type 'q' to quit): ")
    if input_text.lower() == 'q':
        break
    predict_score(input_text, model, tokenizer, bert_model, fasttext_model, mecab, kakasi)
