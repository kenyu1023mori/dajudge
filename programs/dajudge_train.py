import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from itertools import combinations
import MeCab
import pickle  # データのキャッシュ用

# 中間発表時点での進捗に一旦戻してちょい変更
# データパスおよび保存先ディレクトリ設定
file_path = "../../data/count_above_2.csv"
version = "v1.18"
save_model_dir = f"../models/{version}"
os.makedirs(save_model_dir, exist_ok=True)
save_metrics_dir = f"../metrics/{version}"
os.makedirs(save_metrics_dir, exist_ok=True)

# キャッシュ用パス
tokenized_cache_path = "../tokenized_sentences.pkl"

# MeCabを初期化
mecab = MeCab.Tagger("-Owakati")  # MeCabを分かち書きモードで使用

# キャッシュをロードまたは分割して保存
def load_or_tokenize_sentences(sentences):
    if os.path.exists(tokenized_cache_path):
        print("分割済みデータをロード中...")
        with open(tokenized_cache_path, "rb") as f:
            tokenized_sentences = pickle.load(f)
    else:
        print("分割中...（初回のみ実行）")
        tokenized_sentences = [mecab.parse(sentence).strip().split() for sentence in sentences]
        with open(tokenized_cache_path, "wb") as f:
            pickle.dump(tokenized_sentences, f)
    return tokenized_sentences

# 日本語のword2vecモデルを学習データから作成
def train_word2vec_model(tokenized_sentences):
    model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
    model.save("../models/word2vec_dajare.model")
    return model

# 単語間類似度の平均を計算する関数
def get_average_similarity(tokenized_sentence, model):
    vectors = [model.wv[word] for word in tokenized_sentence if word in model.wv]
    if len(vectors) < 2:
        return 0.0
    similarities = [cosine_similarity([v1], [v2])[0][0] for v1, v2 in combinations(vectors, 2)]
    return np.mean(similarities) if similarities else 0.0

# ニューラルネットワークモデル
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

sentences, scores = [], []
with open(file_path, "r", encoding="utf-8") as file:
    for line in file.readlines()[1:]:
        parts = line.strip().split(",")
        if len(parts) == 2:
            sentences.append(parts[0].strip())
            scores.append(float(parts[1].strip()))

# 分割済み文リストをキャッシュからロードまたは新規作成
tokenized_sentences = load_or_tokenize_sentences(sentences)

# word2vecモデルを訓練
w2v_model = train_word2vec_model(tokenized_sentences)

# 特徴量の計算
X = np.array([[get_average_similarity(tokens, w2v_model)] for tokens in tokenized_sentences])
y = np.array(scores)

# モデルを訓練し、評価する関数
def cross_val_train_and_evaluate(X, y, label_name, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_losses, mae_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = DajarePredictor()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        # モデルの訓練
        for epoch in range(30):
            model.train()
            optimizer.zero_grad()
            predictions = model(X_train_tensor)
            loss = criterion(predictions, y_train_tensor)
            loss.backward()
            optimizer.step()
            print(f"Fold {fold+1}/{k}, Epoch {epoch+1}/30, Training Loss: {loss.item()}")

        # 評価
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor)
            mse_loss = criterion(predictions, y_test_tensor).item()
            mae_score = torch.mean(torch.abs(predictions - y_test_tensor)).item()

            mse_losses.append(mse_loss)
            mae_scores.append(mae_score)
            print(f"{label_name} - Fold {fold+1}/{k} - Test MSE Loss: {mse_loss}, Test MAE: {mae_score}")

        # モデルを保存
        torch.save(model.state_dict(), os.path.join(save_model_dir, f"{label_name}_fold_{fold + 1}.pth"))

    avg_mse_loss = np.mean(mse_losses)
    avg_mae_score = np.mean(mae_scores)
    print(f"{label_name} - Average Test MSE Loss: {avg_mse_loss}, Average Test MAE: {avg_mae_score}")

    # 各フォールドごとのlossを棒グラフで保存
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, k + 1), mse_losses, tick_label=[f"Fold {i}" for i in range(1, k + 1)])
    plt.xlabel("Fold")
    plt.ylabel("MSE Loss")
    plt.title(f"{label_name} - MSE Loss per Fold")
    plt.savefig(os.path.join(save_metrics_dir, f"{label_name}_mse_loss_per_fold.png"))
    plt.close()

    # MSE, MAEをテキストファイルに保存
    with open(os.path.join(save_metrics_dir, f"{label_name}_metrics.txt"), "w") as f:
        f.write(f"{label_name} - Test MSE Losses: {mse_losses}\n")
        f.write(f"{label_name} - Test MAE Scores: {mae_scores}\n")
        f.write(f"{label_name} - Average Test MSE Loss: {avg_mse_loss}, Average Test MAE: {avg_mae_score}\n")

# 交差検証
cross_val_train_and_evaluate(X, y, "Dajudge")
