import numpy as np
import torch
import torch.nn as nn
import os
from gensim.models import Word2Vec

# 必要な変数とパスを設定
version = "v1.04"
load_dir = f"/home/group4/evaluate_dajare/models/{version}"
word2vec_model_path = "/home/group4/evaluate_dajare/models/word2vec_dajare.model"

# ニューラルネットワークモデルのクラス定義
class DajarePredictor(nn.Module):
    def __init__(self):
        super(DajarePredictor, self).__init__()
        self.fc1 = nn.Linear(1, 128) # 入力は1次元
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
def get_average_similarity(words, model):
    # モデルに存在する単語のベクトルを取得
    vectors = [model.wv[word] for word in words if word in model.wv]
    if len(vectors) < 2:
        return 0.0  # 単語数が1以下なら類似度は計算できないので0とする

    # 単語ペアごとの類似度を計算
    similarities = []
    for vec1, vec2 in combinations(vectors, 2):
        sim = cosine_similarity([vec1], [vec2])[0][0]
        similarities.append(sim)

    # 類似度の平均を返す
    return np.mean(similarities) if similarities else 0.0

# モデルのロード
models = [[DajarePredictor() for _ in range(5)] for _ in range(3)]
for label_idx in range(3):
    for fold in range(5):
        model_path = os.path.join(load_dir, f"Label_{label_idx+1}_fold_{fold+1}.pth")
        models[label_idx][fold].load_state_dict(torch.load(model_path))
        models[label_idx][fold].eval()

# Word2Vecモデルをロード
w2v_model = Word2Vec.load(word2vec_model_path)

# 入力したダジャレに対してモデルのスコアを出力する関数（各フォールドの平均を使用）
def predict_score(input_text, models, w2v_model):
    tokens = input_text.split()
    input_similarity = get_average_similarity(tokens, w2v_model)
    input_vector = torch.tensor([[input_similarity]], dtype=torch.float32)
    
    with torch.no_grad():
        for label_idx, label_models in enumerate(models, start=1):
            # softmaxを適用せずに予測値を取得
            predictions = [model(input_vector).squeeze() for model in label_models]
            average_prediction = torch.stack(predictions).mean().item()
            predicted_class = round(average_prediction)  # 四捨五入して整数クラスに変換
            predicted_class = max(1, min(predicted_class, 5))  # 1〜5の範囲にクリップ
            print(f"Label {label_idx} - Predicted Score: {predicted_class}")

# ユーザー入力処理
while True:
    input_text = input("Enter a Dajare (or type 'q' to quit): ")
    if input_text.lower() == 'q':
        break
    predict_score(input_text, models, w2v_model)
