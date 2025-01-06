import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import MeCab
import pickle
from transformers import BertTokenizer, BertModel
import pykakasi
import fasttext

# データパスと保存ディレクトリ
file_path = "../../data/count_above_2.csv"
version = "v1.21"
save_model_dir = f"../models/{version}"
os.makedirs(save_model_dir, exist_ok=True)
save_metrics_dir = f"../metrics/{version}"
os.makedirs(save_metrics_dir, exist_ok=True)

# MeCabを初期化
mecab = MeCab.Tagger("-Owakati")  # 分かち書きモード
kakasi = pykakasi.kakasi()  # 音韻解析用

# キャッシュファイル
tokenized_cache_path = "../tokenized_sentences.pkl"

# fastTextモデルパス
fasttext_model_path = "../models/cc.ja.300.bin"
fasttext_model = fasttext.load_model(fasttext_model_path)

# BERTモデルとトークナイザー
bert_model_name = "cl-tohoku/bert-base-japanese"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# 文をBERT埋め込みに変換
def get_bert_embeddings(sentences, tokenizer, model):
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)

# fastText埋め込みを取得
def get_fasttext_embeddings(sentences, model):
    embeddings = []
    for sentence in sentences:
        words = mecab.parse(sentence).strip().split()
        word_embeddings = [model.get_word_vector(word) for word in words]
        if word_embeddings:
            embeddings.append(np.mean(word_embeddings, axis=0))
        else:
            embeddings.append(np.zeros(300))
    return np.array(embeddings)

# 音韻特徴量を生成
def phonetic_features(sentence):
    result = kakasi.convert(sentence)
    romaji = " ".join([item["hepburn"] for item in result])
    length = len(romaji.split())  # 音節数
    vowels = sum(1 for char in romaji if char in "aeiou")  # 母音の数
    consonants = len(romaji.replace(" ", "")) - vowels  # 子音の数
    return [length, vowels, consonants]

# ニューラルネットワークモデル
class DajarePredictor(nn.Module):
    def __init__(self):
        super(DajarePredictor, self).__init__()
        input_size = 768 + 3 + 300  # BERT + 音韻特徴量 + fastText
        self.fc1 = nn.Linear(input_size, 256)
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

# 分割済み文リストをキャッシュからロードまたは新規作成
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

# データ読み込み
sentences, scores = [], []
with open(file_path, "r", encoding="utf-8") as file:
    for line in file.readlines()[1:]:
        parts = line.strip().split(",")
        if len(parts) == 2:
            sentences.append(parts[0].strip())
            scores.append(float(parts[1].strip()))

# 特徴量の生成
bert_embeddings = get_bert_embeddings(sentences, tokenizer, bert_model)
phonetic_features_list = np.array([phonetic_features(sentence) for sentence in sentences])
fasttext_embeddings = get_fasttext_embeddings(sentences, fasttext_model)
X_combined = np.hstack((bert_embeddings, phonetic_features_list, fasttext_embeddings))
y = np.array(scores)

# モデルを訓練し、評価する関数
def cross_val_train_and_evaluate(X, y, label_name, k=5, batch_size=16, epochs=10, accumulation_steps=2):
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

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # モデルの訓練
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += loss.item()
                if (i + 1) % 10 == 0:
                    print(f"Fold {fold+1}/{k}, Epoch {epoch+1}/{epochs}, Step {i+1}, Training Loss: {running_loss / (i + 1)}")

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

# 交差検証の実行
cross_val_train_and_evaluate(X_combined, y, "Dajudge")
