import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

# 学習用データパスやパラメータ設定
file_path = "/home/public/share/MISC/DAJARE/dajare_database_v11.txt"
version = "v1.04"
save_model_dir = f"/home/group4/evaluate_dajare/models/{version}"
os.makedirs(save_model_dir, exist_ok=True)
save_metrics_dir = f"/home/group4/evaluate_dajare/metrics/{version}"
os.makedirs(save_metrics_dir, exist_ok=True)

# 何文使うか
INF = 10e8
limit = INF

# データの読み込みと前処理
dajare_data = []
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()[24:]
    for line in lines:
        if len(dajare_data) >= limit:
            break
        parts = line.strip().split(",")
        if len(parts) >= 5:
            dajare_text = str(parts[1].strip())
            score_1 = float(parts[-4].strip())
            score_2 = float(parts[-3].strip())
            score_3 = float(parts[-2].strip())
            dajare_data.append((dajare_text, score_1, score_2, score_3))

sentences = [dajare[0].split() for dajare in dajare_data]
scores_1 = [dajare[1] for dajare in dajare_data]
scores_2 = [dajare[2] for dajare in dajare_data]
scores_3 = [dajare[3] for dajare in dajare_data]

# Word2Vecモデルのトレーニング
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 訓練後にモデルを保存する
w2v_model.save("/home/group4/evaluate_dajare/models/word2vec_dajare.model")

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

# ニューラルネットワークモデルのクラス定義 (モデルの出力は1つ)
class DajarePredictor(nn.Module):
    def __init__(self):
        super(DajarePredictor, self).__init__()
        self.fc1 = nn.Linear(1, 128)  # 入力を1次元に変更 (コサイン類似度)
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

# MSEとMAEの結果を保存する関数
def save_metrics(metrics, label_name, version):
    result_file = os.path.join(save_metrics_dir, f"{label_name}_metrics.txt")
    
    with open(result_file, "w") as file:
        file.write(f"{label_name} - Test MSE Losses: {metrics['mse']}\n")
        file.write(f"{label_name} - Test MAE Scores: {metrics['mae']}\n")
        file.write(f"{label_name} - Average Test MSE Loss: {np.mean(metrics['mse'])}, Average Test MAE: {np.mean(metrics['mae'])}\n")

# 結果をプロットして保存する関数
def plot_metrics(metrics, label_name, version, y_mse_range=(0, 1), y_mae_range=(0, 1)):
    # 平均MSEと平均MAE
    avg_mse = np.mean(metrics['mse'])
    avg_mae = np.mean(metrics['mae'])
    
    # 棒グラフ
    plt.figure(figsize=(12, 6))

    # MSE Lossのプロット
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(metrics['mse']) + 1), metrics['mse'], color='lightsteelblue', alpha=0.7)
    plt.axhline(y=avg_mse, color='red', linestyle='--', label=f"Average MSE: {avg_mse:.3f}")
    plt.ylim(y_mse_range)
    plt.xlabel("Fold")
    plt.ylabel("MSE Loss")
    plt.title(f"{label_name} MSE Loss per Fold")
    plt.legend()

    # MAEのプロット
    plt.subplot(1, 2, 2)
    plt.bar(range(1, len(metrics['mae']) + 1), metrics['mae'], color='lightsteelblue', alpha=0.7)
    plt.axhline(y=avg_mae, color='red', linestyle='--', label=f"Average MAE: {avg_mae:.3f}")
    plt.ylim(y_mae_range)
    plt.xlabel("Fold")
    plt.ylabel("MAE")
    plt.title(f"{label_name} MAE per Fold")
    plt.legend()

    # グラフの保存
    bar_plot_file = os.path.join(save_metrics_dir, f"{label_name}_metrics_bar_plot.png")
    plt.savefig(bar_plot_file)
    plt.close()

# 分割交差検証とモデルの保存を行う関数
def cross_val_train_and_evaluate(X, y, label_name, version, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_losses, mae_scores = [], []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        model = DajarePredictor()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        epochs, batch_size = 30, 64
        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(X_train_tensor.size()[0])
            for i in range(0, X_train_tensor.size()[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        # 評価
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor)
            mse_loss = criterion(predictions, y_test_tensor).item()
            mae = torch.mean(torch.abs(torch.round(predictions) - y_test_tensor)).item()
            
            mse_losses.append(mse_loss)
            mae_scores.append(mae)
            print(f"{label_name} - Fold {fold+1}/{k} - Test MSE Loss: {mse_loss}, Test MAE: {mae}")

        torch.save(model.state_dict(), os.path.join(save_model_dir, f"{label_name}_fold_{fold+1}.pth"))

    print(f"{label_name} - Average Test MSE Loss: {np.mean(mse_losses)}, Average Test MAE: {np.mean(mae_scores)}")
    
    # メトリクスの保存とプロット作成
    metrics = {"mse": mse_losses, "mae": mae_scores}
    save_metrics(metrics, label_name, version)
    plot_metrics(metrics, label_name, version)
    print(f"{label_name} - Average Test MSE Loss: {np.mean(mse_losses)}, Average Test MAE: {np.mean(mae_scores)}")

# 文ごとに単語間類似度の平均を特徴量として取得
X = np.array([[get_average_similarity(sentence, w2v_model)] for sentence in sentences])  # Xを(サンプル数, 1)に整形

# モデルのトレーニングと評価を実行
cross_val_train_and_evaluate(X, scores_1, "Label_1", version)
cross_val_train_and_evaluate(X, scores_2, "Label_2", version)
cross_val_train_and_evaluate(X, scores_3, "Label_3", version)
