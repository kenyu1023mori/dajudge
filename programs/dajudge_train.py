import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample
import matplotlib.pyplot as plt
import MeCab
import pickle
import pandas as pd
from transformers import BertJapaneseTokenizer, BertModel
import pykakasi
import fasttext

# データパスと保存ディレクトリ
file_path = "../../data/evenly_after_shareka.csv"
version = "v2.06"
save_model_dir = f"../models/{version}"
os.makedirs(save_model_dir, exist_ok=True)
save_metrics_dir = f"../metrics/{version}"
os.makedirs(save_metrics_dir, exist_ok=True)

# MeCabと音韻解析の初期化
mecab = MeCab.Tagger("-Owakati")
kakasi = pykakasi.kakasi()

# キャッシュファイルとfastTextモデル
tokenized_cache_path = "../tokenized_sentences.pkl"
fasttext_model_path = "../models/cc.ja.300.bin"
fasttext_model = fasttext.load_model(fasttext_model_path)

# BERTモデルとトークナイザー
bert_model_name = "cl-tohoku/bert-base-japanese"
tokenizer = BertJapaneseTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# 特徴量生成関数
def get_bert_embeddings(sentences, tokenizer, model, batch_size=16):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        inputs = tokenizer(batch_sentences, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

def get_fasttext_embeddings(sentences, model):
    embeddings = []
    for sentence in sentences:
        words = mecab.parse(sentence).strip().split()
        word_embeddings = [model.get_word_vector(word) for word in words]
        embeddings.append(np.mean(word_embeddings, axis=0) if word_embeddings else np.zeros(300))
    return np.array(embeddings)

def phonetic_features(sentence):
    result = kakasi.convert(sentence)
    romaji = " ".join([item["hepburn"] for item in result])
    length = len(romaji.split())
    vowels = sum(1 for char in romaji if char in "aeiou")
    consonants = len(romaji.replace(" ", "")) - vowels
    return [length, vowels, consonants]

# ニューラルネットワークモデル
class DajarePredictor(nn.Module):
    def __init__(self):
        super(DajarePredictor, self).__init__()
        input_size = 768 + 3 + 300
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return self.sigmoid(x)

# データ読み込みと前処理
data = pd.read_csv(file_path)
sentences = data['dajare'].astype(str).tolist()
scores = data['score'].tolist()

# データバランス調整（オーバーサンプリング）
df = pd.DataFrame({'sentence': sentences, 'score': scores})
balanced_df = df.groupby('score', group_keys=False).apply(lambda x: resample(x, replace=True, n_samples=3000))
sentences = balanced_df['sentence'].tolist()
scores = balanced_df['score'].tolist()

# 特徴量の生成
bert_embeddings = get_bert_embeddings(sentences, tokenizer, bert_model)
phonetic_features_list = np.array([phonetic_features(sentence) for sentence in sentences])
fasttext_embeddings = get_fasttext_embeddings(sentences, fasttext_model)
X_combined = np.hstack((bert_embeddings, phonetic_features_list, fasttext_embeddings))

# 特徴量の正規化
X_combined = (X_combined - np.mean(X_combined, axis=0)) / np.std(X_combined, axis=0)
y = np.array(scores) / 5.0  # スコアを0～1に正規化

# モデルを訓練し、評価する関数
def cross_val_train_and_evaluate(X, y, label_name, k=5, batch_size=16, epochs=20, learning_rate=0.0001):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_losses, mae_scores = [], []
    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = DajarePredictor()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.HuberLoss()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                predictions = model(X_test_tensor)
                mse_loss = nn.MSELoss()(predictions, y_test_tensor).item()
                mae_score = mean_absolute_error(y_test_tensor.numpy(), predictions.numpy())
                mse_losses.append(mse_loss)
                mae_scores.append(mae_score)
                all_predictions.extend(predictions.numpy().flatten())
                print(f"Fold {fold+1}, Epoch {epoch+1}, MSE Loss: {mse_loss}, MAE: {mae_score}")

        torch.save(model.state_dict(), os.path.join(save_model_dir, f"{label_name}_fold_{fold + 1}.pth"))

    # 結果を可視化
    avg_mse = np.mean(mse_losses)
    avg_mae = np.mean(mae_scores)
    print(f"Average MSE Loss: {avg_mse}, Average MAE: {avg_mae}")

cross_val_train_and_evaluate(X_combined, y, "Dajare")
