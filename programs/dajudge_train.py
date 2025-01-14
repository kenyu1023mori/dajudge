import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import MeCab
import pickle
import pandas as pd
from transformers import BertJapaneseTokenizer, BertModel
import pykakasi
import fasttext

# データパスと保存ディレクトリ
file_path = "../../data/evenly_after_shareka.csv"
version = "v2.12"
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
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation

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
        # ToDo: ここおかしいかも
        return x
# データ読み込みと前処理
data = pd.read_csv(file_path)
sentences = data['dajare'].astype(str).tolist()
scores = data['score'].tolist()

# 特徴量の生成
bert_embeddings = get_bert_embeddings(sentences, tokenizer, bert_model)
phonetic_features_list = np.array([phonetic_features(sentence) for sentence in sentences])
fasttext_embeddings = get_fasttext_embeddings(sentences, fasttext_model)
X_combined = np.hstack((bert_embeddings, phonetic_features_list, fasttext_embeddings))

# 特徴量の正規化
X_combined = (X_combined - np.mean(X_combined, axis=0)) / np.std(X_combined, axis=0)
y = np.array(scores) / 5.0  # スコアを0～1に正規化

# データセットの分割
X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# モデルを訓練し、評価する関数
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, label_name, batch_size=16, epochs=20, learning_rate=0.0001):
    model = DajarePredictor()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # ToDo: 損失関数はHuber Loss、要検討
    criterion = nn.HuberLoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    val_losses = []
    val_maes = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_mse_loss = nn.MSELoss()(val_predictions, y_val_tensor).item()
            val_mae_score = mean_absolute_error(y_val_tensor.numpy(), val_predictions.numpy())
            val_losses.append(val_mse_loss)
            val_maes.append(val_mae_score)
            print(f"Epoch {epoch+1}, Training Loss: {epoch_train_loss}, Validation MSE Loss: {val_mse_loss}, Validation MAE: {val_mae_score}")

    torch.save(model.state_dict(), os.path.join(save_model_dir, f"{label_name}.pth"))

    # テストデータで評価
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_mse_loss = nn.MSELoss()(test_predictions, y_test_tensor).item()
        test_mae_score = mean_absolute_error(y_test_tensor.numpy(), test_predictions.numpy())
        print(f"Test MSE Loss: {test_mse_loss}, Test MAE: {test_mae_score}")

    # 予測スコアの分布をヒストグラムで保存
    plt.figure(figsize=(12, 6))
    plt.hist(test_predictions.numpy(), bins=50, edgecolor='k', color='skyblue', alpha=0.7, label='Predicted')
    plt.hist(y_test_tensor.numpy(), bins=50, edgecolor='k', color='orange', alpha=0.5, label='True')

    # x軸の目盛りを0.1刻みに設定
    min_score = min(test_predictions.min().item(), y_test_tensor.min().item())  # 最小予測値
    max_score = max(test_predictions.max().item(), y_test_tensor.max().item())  # 最大予測値
    ticks = np.arange(np.floor(min_score * 10) / 10, np.ceil(max_score * 10) / 10 + 0.1, 0.1)
    plt.xticks(ticks)  # 目盛りを設定

    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title(f"{label_name} - Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_metrics_dir, f"{label_name}_score_distribution.png"))
    plt.close()

    # 損失関数の推移を棒グラフに出力
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='orange')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation MSE Loss', color='skyblue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_metrics_dir, f"{label_name}_loss.png"))
    plt.close()

    # 損失関数の推移をテキストとして保存
    with open(os.path.join(save_metrics_dir, f"{label_name}_loss.txt"), "w") as f:
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f"Epoch {epoch}: Training Loss: {train_loss}, Validation MSE Loss: {val_loss}, Validation MAE: {val_maes[epoch-1]}\n")

train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, "Dajare")
