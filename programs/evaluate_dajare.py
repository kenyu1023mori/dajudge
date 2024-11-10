import numpy as np
import torch
import torch.nn as nn
import os
from gensim.models import Word2Vec

# 必要な変数とパスを設定
version = "v1.01"
load_dir = f"/home/group4/evaluate_dajare/models/{version}"
word2vec_model_path = "/home/group4/evaluate_dajare/models/word2vec_dajare.model"  # Word2Vecモデルのパスを指定

# ニューラルネットワークモデルのクラス定義
class DajarePredictor(nn.Module):
    def __init__(self):
        super(DajarePredictor, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# ダジャレをベクトル化する関数
def get_average_vector(words, model, vector_size=100):
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

# モデルのロード
models = [[DajarePredictor() for _ in range(5)] for _ in range(3)]
for label_idx in range(3):
    for fold in range(5):
        model_path = os.path.join(load_dir, f"Label_{label_idx+1}_fold_{fold+1}.pth")
        models[label_idx][fold].load_state_dict(torch.load(model_path))
        models[label_idx][fold].eval()

# Word2Vecモデルをロード
w2v_model = Word2Vec.load(word2vec_model_path)

# 入力したダジャレに対してモデルのスコアを出力する関数（各フォルドの平均を使用）
def predict_score(input_text, models, w2v_model):
    tokens = input_text.split()
    input_vector = torch.tensor(get_average_vector(tokens, w2v_model)).float().view(1, -1)
    with torch.no_grad():
        for label_idx, label_models in enumerate(models, start=1):
            predictions = [model(input_vector).item() for model in label_models]
            average_prediction = torch.round(torch.tensor(predictions).mean()).item()
            print(f"Label {label_idx} - Predicted Score: {average_prediction}")

# ユーザー入力処理ループ
while True:
    input_text = input("Enter a Dajare (or type 'q' to quit): ")
    if input_text.lower() == 'q':
        break
    predict_score(input_text, models, w2v_model)
