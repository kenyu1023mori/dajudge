import numpy as np
import torch
import torch.nn as nn
import os
from gensim.models import Word2Vec
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import MeCab

# 中間発表時点での進捗に一旦戻してちょい変更
# 必要な変数とパスを設定
version = "v1.18"
load_dir = f"../models/{version}"
word2vec_model_path = "../models/word2vec_dajare.model"

# MeCabの設定
mecab = MeCab.Tagger("-Owakati")  # 単語を分かち書き形式で取得

# ニューラルネットワークモデルのクラス定義
class DajarePredictor(nn.Module):
    def __init__(self):
        super(DajarePredictor, self).__init__()
        self.fc1 = nn.Linear(1, 128)
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

# ダジャレをベクトル化する関数
def get_average_similarity(words, model):
    vectors = [model.wv[word] for word in words if word in model.wv]
    if len(vectors) < 2:
        return 0.0
    similarities = [cosine_similarity([v1], [v2])[0][0] for v1, v2 in combinations(vectors, 2)]
    return np.mean(similarities) if similarities else 0.0

# モデルのロード
models = [DajarePredictor() for _ in range(5)]
for fold in range(5):
    model_path = os.path.join(load_dir, f"Dajudge_fold_{fold+1}.pth")
    models[fold].load_state_dict(torch.load(model_path))
    models[fold].eval()

# Word2Vecモデルをロード
w2v_model = Word2Vec.load(word2vec_model_path)

# 入力したダジャレに対してモデルのスコアを出力する関数
def predict_score(input_text, models, w2v_model, mecab):
    tokenized_text = mecab.parse(input_text).strip()
    tokens = tokenized_text.split()
    input_similarity = get_average_similarity(tokens, w2v_model)
    input_vector = torch.tensor([[input_similarity]], dtype=torch.float32)

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
    predict_score(input_text, models, w2v_model, mecab)
