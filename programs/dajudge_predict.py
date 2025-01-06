import numpy as np
import torch
import torch.nn as nn
import os
from transformers import BertModel, BertTokenizer
import MeCab

# 必要な変数とパスを設定
version = "v1.20"
load_dir = f"../models/{version}"
bert_model_name = "cl-tohoku/bert-base-japanese"

# BERTモデルとトークナイザーの初期化
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# MeCabの設定
mecab = MeCab.Tagger("-Owakati")  # 単語を分かち書き形式で取得

# ニューラルネットワークモデルのクラス定義
class DajarePredictor(nn.Module):
    def __init__(self):
        super(DajarePredictor, self).__init__()
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
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
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# モデルのロード
models = [DajarePredictor() for _ in range(5)]
for fold in range(5):
    model_path = os.path.join(load_dir, f"Dajudge_fold_{fold+1}.pth")
    models[fold].load_state_dict(torch.load(model_path))
    models[fold].eval()

# 入力したダジャレに対してモデルのスコアを出力する関数
def predict_score(input_text, models, tokenizer, bert_model):
    input_embedding = get_bert_embedding(input_text, tokenizer, bert_model)
    input_vector = torch.tensor([input_embedding], dtype=torch.float32)

    with torch.no_grad():
        predictions = [model(input_vector).squeeze() for model in models]
        average_prediction = torch.stack(predictions).mean().item()
        predicted_class = round(average_prediction)
        predicted_class = max(1, min(predicted_class, 5))

        print(f"Predicted Score: {predicted_class}")

# ユーザー入力処理
while True:
    input_text = input("Enter a Dajare (or type 'q' to quit): ")
    if input_text.lower() == 'q':
        break
    predict_score(input_text, models, tokenizer, bert_model)
