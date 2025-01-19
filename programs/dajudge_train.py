import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import MeCab
import pandas as pd
from transformers import BertJapaneseTokenizer, BertModel
import fasttext
import optuna

# データパスと保存ディレクトリ
file_path = "../../data/final/dajare_dataset.csv"
version = "v3.10"
save_model_dir = f"../models/{version}"
os.makedirs(save_model_dir, exist_ok=True)
save_metrics_dir = f"../metrics/{version}"
os.makedirs(save_metrics_dir, exist_ok=True)

# MeCabの初期化
mecab = MeCab.Tagger("-Owakati")

# fastTextモデル
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

# 日本語のストップワード
japanese_stop_words = set([
    "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ", "ある", "いる", 
    "も", "する", "から", "な", "こと", "として", "い", "や", "れる", "など", "なっ", "ない", 
    "この", "ため", "その", "あっ", "よう", "また", "もの", "という", "あり", "まで", "られ", 
    "なる", "へ", "か", "だ", "これ", "によって", "により", "おり", "より", "による", "ず", 
    "なり", "られる", "、", "。", "「", "」", "！", "？", "〜", "ー", ",", "."
])

def get_fasttext_embeddings(sentences, model):
    embeddings = []
    for sentence in sentences:
        words = mecab.parse(sentence).strip().split()
        # fastTextのembeddingではストップワードを除外して考える
        word_embeddings = [model.get_word_vector(word) for word in words if word not in japanese_stop_words]
        embeddings.append(np.mean(word_embeddings, axis=0) if word_embeddings else np.zeros(300))
    return np.array(embeddings)

# 特徴量の生成
def generate_features(sentences):
    bert_embeddings = get_bert_embeddings(sentences, tokenizer, bert_model)
    fasttext_embeddings = get_fasttext_embeddings(sentences, fasttext_model)

    # 特徴量の結合
    X_combined = np.hstack((bert_embeddings, fasttext_embeddings))
    return X_combined

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
X_combined = generate_features(sentences)

# 特徴量の標準化
X_combined = (X_combined - np.mean(X_combined, axis=0)) / np.std(X_combined, axis=0)
# スコアの正規化 (x-x_min)/(x_max-x_min)
y = (np.array(scores) - 1.0) / 3.6

# データセットの分割
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# 5分割交差検証の設定
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Optunaの目的関数
def objective(trial):
    hidden_sizes = [
        trial.suggest_int("hidden_size1", 128, 512),
        trial.suggest_int("hidden_size2", 64, 256),
        trial.suggest_int("hidden_size3", 32, 128),
        trial.suggest_int("hidden_size4", 16, 64)
    ]
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    epochs = trial.suggest_int("epochs", 10, 100)

    model = DajarePredictor(input_size=1068, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # RMSE損失

    # データを訓練データと検証データに分割
    X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(X_train, y_train, test_size=0.2, random_state=trial.suggest_int("split_seed", 0, 10000))

    X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_fold, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_fold, dtype=torch.float32).view(-1, 1)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)  # RMSE損失を計算
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

# 交差検証の結果を保存するリスト
all_train_losses = []
all_val_losses = []
all_val_mae_losses = []
best_fold = None
best_val_rmse = float('inf')

for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}")
    
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # 各フォールドごとに新しいモデル、オプティマイザ、損失関数を定義
    model = DajarePredictor(input_size=1068, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)  # Update input size
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # RMSE損失

    X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_fold, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_fold, dtype=torch.float32).view(-1, 1)

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
            loss = criterion(predictions, targets)  # RMSE損失を計算
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

    # 最も性能の良いモデルを選択
    if val_rmse_loss < best_val_rmse:
        best_val_rmse = val_rmse_loss
        best_fold = fold + 1
        torch.save(model.state_dict(), os.path.join(save_model_dir, "Dajare_best.pth"))

    # 損失関数の推移を棒グラフに出力（RMSE）
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='orange')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation RMSE', color='skyblue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation RMSE Over Epochs (Fold {fold + 1})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_metrics_dir, f"Dajare_rmse_loss_fold{fold + 1}.png"))
    plt.close()

    # 損失関数の推移を棒グラフに出力（MAE）
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), val_mae_losses, label='Validation MAE', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Validation MAE Over Epochs (Fold {fold + 1})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_metrics_dir, f"Dajare_mae_loss_fold{fold + 1}.png"))
    plt.close()

    # 損失関数の推移をテキストとして保存
    with open(os.path.join(save_metrics_dir, f"Dajare_loss_fold{fold + 1}.txt"), "a") as f:
        for epoch, (train_loss, val_loss, val_mae_loss) in enumerate(zip(train_losses, val_losses, val_mae_losses), 1):
            f.write(f"Epoch {epoch}: Training Loss: {train_loss}, Validation RMSE: {val_loss}, Validation MAE: {val_mae_loss}\n")

    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_val_mae_losses.append(val_mae_losses)

print(f"Best fold: {best_fold}")

# 最も性能の良いモデルをロード
model.load_state_dict(torch.load(os.path.join(save_model_dir, "Dajare_best.pth"), weights_only=True))

# テストデータで評価
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    test_predictions = model(X_test_tensor)
    test_rmse_loss = mean_squared_error(y_test_tensor.numpy(), test_predictions.numpy(), squared=False)
    test_mae_loss = mean_absolute_error(y_test_tensor.numpy(), test_predictions.numpy())  # MAEを計算
    print(f"Test RMSE: {test_rmse_loss}, Test MAE: {test_mae_loss}")

# スケールを1～5に戻す
test_predictions = test_predictions * 4 + 1
y_test_tensor = y_test_tensor * 4 + 1

# テストデータのインデックスを取得
_, test_indices = train_test_split(data.index, test_size=0.2, random_state=42)

# テストデータのDataFrameを作成し、予測スコアを追加
test_data = data.loc[test_indices].copy()
test_data['predict'] = test_predictions.numpy()

# テストデータと予測スコアを保存
test_data.to_csv(os.path.join(save_metrics_dir, "test_predictions.csv"), index=False)

# 回帰モデルの出力をカテゴリに変換
test_predictions_rounded = np.round(test_predictions.numpy()).astype(int)
y_test_rounded = np.round(y_test_tensor.numpy()).astype(int)

# 面白い/面白くないの分類
y_true = (y_test_rounded >= 3).astype(int).flatten()
y_pred = (test_predictions_rounded >= 3).astype(int).flatten()

# 評価指標の計算
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# y_trueと y_predの数を計算
y_true_count = np.bincount(y_true)
y_pred_count = np.bincount(y_pred)

# 評価指標の保存
with open(os.path.join(save_metrics_dir, "Dajare_loss.txt"), "a") as f:
    f.write(f"Test RMSE: {test_rmse_loss}, Test MAE: {test_mae_loss}\n")
    f.write(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}\n")
    f.write(f"y_true counts: {y_true_count.tolist()}\n")
    f.write(f"y_pred counts: {y_pred_count.tolist()}\n")

# 予測スコアの分布をヒストグラムで保存
plt.figure(figsize=(12, 6))
plt.hist(test_predictions.numpy(), bins=50, edgecolor='k', color='skyblue', alpha=0.7, label='Predicted')
plt.hist(y_test_tensor.numpy(), bins=50, edgecolor='k', color='orange', alpha=0.5, label='True')

min_score = min(test_predictions.min().item(), y_test_tensor.min().item())
max_score = max(test_predictions.max().item(), y_test_tensor.max().item())  # 修正箇所
ticks = np.arange(np.floor(min_score * 10) / 10, np.ceil(max_score * 10) / 10 + 0.1)
plt.xticks(ticks)

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
