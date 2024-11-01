import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# ダジャレ本文と 3 人のスコア (scores_1, 2, 3) で、別々に学習させる。
# word2vec, DNN, Adam, 活性化関数は ReLU
# 評価は MSE と MAE

# epochs = 30
# batch_size = 64

"""
Label 1 - Test MSE Loss: 0.47937116026878357
Label 1 - Test MAE (rounded): 0.4788805842399597
Label 2 - Test MSE Loss: 0.5860065221786499
Label 2 - Test MAE (rounded): 0.5835074782371521
Label 3 - Test MSE Loss: 0.21585750579833984
Label 3 - Test MAE (rounded): 0.21514925360679626
"""

# group4 の csv はまだ変更加えそうなのでホームから
file_path = "/home/public/share/MISC/DAJARE/dajare_database_v11.txt"
pun_data = []

INF = 10e8
# 学習に使うデータ数
lim = INF

# 学習と評価をするか
is_train_and_evaluate = True

try:
    with open(file_path, "r", encoding="utf-8") as file:
        # ダジャレデータは 25 行目から
        lines = file.readlines()[24:]
        
        for line in lines:
            if len(pun_data) >= lim:
                break
            parts = line.strip().split(",")
            
            if len(parts) >= 5:
                pun_text = str(parts[1].strip())
                score_1 = float(parts[-4].strip())
                score_2 = float(parts[-3].strip())
                score_3 = float(parts[-2].strip())
                pun_data.append((pun_text, score_1, score_2, score_3))

except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

# 駄洒落をトークン化
sentences = [pun[0].split() for pun in pun_data if isinstance(pun[0], str)]
scores_1 = [pun[1] for pun in pun_data]
scores_2 = [pun[2] for pun in pun_data]
scores_3 = [pun[3] for pun in pun_data]

# Word2Vecモデルのトレーニング
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 駄洒落の平均ベクトルを取得する関数
def get_average_vector(words, model, vector_size=100):
    vectors = [model.wv[word] for word in words if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

# 全ての駄洒落をベクトル化
X = np.array([get_average_vector(sentence, w2v_model) for sentence in sentences])

# ニューラルネットワークモデルの構築 (DajarePredictor) のクラス定義
class DajarePredictor(nn.Module):
    def __init__(self):
        super(DajarePredictor, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # 出力層は1ユニット (連続値の予測)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 学習と評価を行う関数、モデルの保存も
def train_and_evaluate_model(X, y, label_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DajarePredictor()
    criterion = nn.L1Loss()  # 平均絶対誤差 (MAE) を重視
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # データをTensorに変換
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # モデルの学習
    epochs = 30
    batch_size = 64

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_tensor.size()[0])
        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # print(f"{label_name} - Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # モデルの評価
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        rounded_predictions = torch.round(predictions)  # 1～5の範囲に四捨五入
        test_loss = criterion(predictions, y_test_tensor).item()
        mae = torch.mean(torch.abs(rounded_predictions - y_test_tensor)).item()
        print(f"{label_name} - Test MSE Loss: {test_loss}")
        print(f"{label_name} - Test MAE (rounded): {mae}")
        
    # モデルをファイルに保存
    torch.save(model.state_dict(), f'/home/group4/evaluate_puns/models/{label_name}_model.pth')

# 各ラベルでモデルの学習と評価を行う
if is_train_and_evaluate:
    train_and_evaluate_model(X, scores_1, "Label_1")
    train_and_evaluate_model(X, scores_2, "Label_2")
    train_and_evaluate_model(X, scores_3, "Label_3")


# 入力したダジャレに対してモデルのスコアを出力する関数
def predict_score(input_text, models, w2v_model):
    # ダジャレをトークン化
    tokens = input_text.split()
    # 平均ベクトルを取得
    input_vector = torch.tensor(get_average_vector(tokens, w2v_model)).float().view(1, -1)

    # 各モデルでスコアを予測
    with torch.no_grad():
        for i, model in enumerate(models, start=1):
            prediction = model(input_vector)
            rounded_prediction = torch.round(prediction).item()
            print(f"Model {i} - Predicted Score: {rounded_prediction:.1f}")

# モデルをリストにまとめる
models = [
    DajarePredictor(),  # ラベル1のモデル
    DajarePredictor(),  # ラベル2のモデル
    DajarePredictor(),  # ラベル3のモデル
]

# 学習済みパラメータをロード
for i in range(3):
    models[i].load_state_dict(torch.load(f'/home/group4/evaluate_puns/models/Label_{i+1}_model.pth'))  # 推論時にロード

# 繰り返し入力処理
while True:
    input_text = input("Enter a Dajare (or type 'q' to quit): ")
    if input_text.lower() == 'q':
        break
    predict_score(input_text, models, w2v_model)
