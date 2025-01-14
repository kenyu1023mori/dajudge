import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 変更
import matplotlib.pyplot as plt
import MeCab
import pickle
import pandas as pd
from transformers import BertJapaneseTokenizer, BertModel
import pykakasi
import fasttext
import optuna  # Optunaをインポート
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # 追加

# データパスと保存ディレクトリ
file_path = "../../data/final/dajare_dataset.csv"
version = "v3.00"
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
bert_model_name = "cl-tohoku/bert-base-japanese-v3"
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
    def __init__(self, input_size, hidden_sizes, dropout_rate):
        super(DajarePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

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
y = (np.array(scores) - 1) / 4.0  # スコアを1～5から0～1に正規化

# データセットの分割
X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Optunaの目的関数

def objective(trial):
    hidden_sizes = [
        trial.suggest_int("hidden_size1", 128, 512),  # 範囲を調整
        trial.suggest_int("hidden_size2", 64, 256),   # 範囲を調整
        trial.suggest_int("hidden_size3", 32, 128),   # 範囲を調整
        trial.suggest_int("hidden_size4", 16, 64)     # 範囲を調整
    ]
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)  # 範囲を調整
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_int("batch_size", 16, 128)  # 範囲を調整
    epochs = trial.suggest_int("epochs", 10, 100)  # 範囲を調整

    model = DajarePredictor(input_size=1071, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # 損失関数をMSEに変更

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = torch.sqrt(criterion(predictions, targets))  # RMSEを計算
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_tensor)
        val_rmse_loss = mean_squared_error(y_val_tensor.numpy(), val_predictions.numpy(), squared=False)

    return val_rmse_loss

# Optunaによるハイパーパラメータ探索
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# 最適なハイパーパラメータでモデルを再訓練
best_params = study.best_params
print("Best parameters:", best_params)

hidden_sizes = [
    best_params["hidden_size1"],
    best_params["hidden_size2"],
    best_params["hidden_size3"],
    best_params["hidden_size4"]
]
dropout_rate = best_params["dropout_rate"]
learning_rate = best_params["learning_rate"]
batch_size = best_params["batch_size"]
epochs = best_params["epochs"]

"""
# 手動でパラメータを指定
hidden_sizes = [761, 368, 94, 112]
dropout_rate = 0.4567045577962901
learning_rate = 0.0005650662190333565
batch_size = 47
epochs = 47
"""

model = DajarePredictor(input_size=1071, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()  # 損失関数をMSEに変更

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
val_mae_losses = []  # MAEの損失関数の推移を保存するリスト

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = torch.sqrt(criterion(predictions, targets))  # RMSEを計算
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_tensor)
        val_rmse_loss = mean_squared_error(y_val_tensor.numpy(), val_predictions.numpy(), squared=False)
        val_mae_loss = mean_absolute_error(y_val_tensor.numpy(), val_predictions.numpy())  # MAEを計算
        val_losses.append(val_rmse_loss)
        val_mae_losses.append(val_mae_loss)
        print(f"Epoch {epoch+1}, Training Loss: {epoch_train_loss}, Validation RMSE: {val_rmse_loss}, Validation MAE: {val_mae_loss}")

torch.save(model.state_dict(), os.path.join(save_model_dir, "Dajare.pth"))

# テストデータで評価
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_rmse_loss = mean_squared_error(y_test_tensor.numpy(), test_predictions.numpy(), squared=False)
    test_mae_loss = mean_absolute_error(y_test_tensor.numpy(), test_predictions.numpy())  # MAEを計算
    print(f"Test RMSE: {test_rmse_loss}, Test MAE: {test_mae_loss}")

# スケールを1～5に戻す
test_predictions = test_predictions * 4 + 1
y_test_tensor = y_test_tensor * 4 + 1

# テストデータの保存
test_indices = X_temp.shape[0] // 2 + np.arange(X_test_tensor.shape[0])
test_data = data.iloc[test_indices].copy()
test_data.loc[:, 'predict'] = test_predictions.numpy()
test_data.to_csv(os.path.join(save_metrics_dir, "test_data_with_predictions.csv"), index=False)

# 面白い/面白くないの分類
# y_true = (y_test_tensor.numpy() >= 3.0).astype(int)
# y_pred = (test_predictions.numpy() >= 3.0).astype(int)

# 評価指標の計算
# accuracy = accuracy_score(y_true, y_pred)
# precision = precision_score(y_true, y_pred)
# recall = recall_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)

# 評価指標の保存
with open(os.path.join(save_metrics_dir, "Dajare_loss.txt"), "a") as f:
    f.write(f"Test RMSE: {test_rmse_loss}, Test MAE: {test_mae_loss}\n")
    # f.write(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}\n")

# 予測スコアの分布をヒストグラムで保存
plt.figure(figsize=(12, 6))
plt.hist(test_predictions.numpy(), bins=50, edgecolor='k', color='skyblue', alpha=0.7, label='Predicted')
plt.hist(y_test_tensor.numpy(), bins=50, edgecolor='k', color='orange', alpha=0.5, label='True')

# x軸の目盛りを0.1刻みに設定
min_score = min(test_predictions.min().item(), y_test_tensor.min().item())  # 最小予測値
max_score = max(test_predictions.max().item(), y_test_tensor.max().item())  # 最大予測値
ticks = np.arange(np.floor(min_score * 10) / 10, np.ceil(max_score * 10) / 10 + 0.1)
plt.xticks(ticks)  # 目盛りを設定

plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Dajare - Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_metrics_dir, "Dajare_score_distribution.png"))
plt.close()

# 損失関数の推移を棒グラフに出力（RMSE）
plt.figure(figsize=(12, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='orange')
plt.plot(range(1, epochs + 1), val_losses, label='Validation RMSE', color='skyblue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation RMSE Over Epochs')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_metrics_dir, "Dajare_rmse_loss.png"))
plt.close()

# 損失関数の推移を棒グラフに出力（MAE）
plt.figure(figsize=(12, 6))
plt.plot(range(1, epochs + 1), val_mae_losses, label='Validation MAE', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation MAE Over Epochs')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_metrics_dir, "Dajare_mae_loss.png"))
plt.close()

# 損失関数の推移をテキストとして保存
with open(os.path.join(save_metrics_dir, "Dajare_loss.txt"), "a") as f:
    for epoch, (train_loss, val_loss, val_mae_loss) in enumerate(zip(train_losses, val_losses, val_mae_losses), 1):
        f.write(f"Epoch {epoch}: Training Loss: {train_loss}, Validation RMSE: {val_loss}, Validation MAE: {val_mae_loss}\n")
