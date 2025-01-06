import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import BertModel, BertTokenizer
import pickle
import MeCab

# データパスおよび保存先
file_path = "../../data/count_above_2.csv"
version = "v1.20"
save_model_dir = f"../models/{version}"
os.makedirs(save_model_dir, exist_ok=True)
save_metrics_dir = f"../metrics/{version}"
os.makedirs(save_metrics_dir, exist_ok=True)

# キャッシュ用パス
tokenized_cache_path = "../tokenized_sentences.pkl"

# GPU利用設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MeCabを初期化
mecab = MeCab.Tagger("-Owakati")

# 分割済みデータのロードまたは作成
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

# BERT埋め込みを取得（バッチ処理対応）
def get_bert_embeddings(sentences, tokenizer, model, batch_size=16):
    model.eval()
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        inputs = tokenizer(batch_sentences, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# ニューラルネットワークモデル
class DajarePredictor(nn.Module):
    def __init__(self):
        super(DajarePredictor, self).__init__()
        self.fc1 = nn.Linear(768, 128)
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

# モデルの訓練と評価
def cross_val_train_and_evaluate(X, y, label_name, k=5, batch_size=16, epochs=10, accumulation_steps=2, verbose=True):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics = {"mse": [], "mae": [], "r2": []}

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = DajarePredictor().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
        criterion = nn.MSELoss()

        # データセットとローダー
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                                                       torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 訓練ループ
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 勾配クリッピング
                    optimizer.step()

                running_loss += loss.item()

            scheduler.step()
            if verbose:
                print(f"Fold {fold+1}/{k}, Epoch {epoch+1}/{epochs}, Training Loss: {running_loss / len(train_loader)}")

        # テスト評価
        model.eval()
        with torch.no_grad():
            predictions = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            metrics["mse"].append(mse)
            metrics["mae"].append(mae)
            metrics["r2"].append(r2)

            print(f"{label_name} - Fold {fold+1}/{k} - MSE: {mse}, MAE: {mae}, R^2: {r2}")

        # モデル保存
        torch.save(model.state_dict(), os.path.join(save_model_dir, f"{label_name}_fold_{fold + 1}.pth"))

    # 平均とプロット
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(f"{label_name} - Average Metrics: {avg_metrics}")

    plt.figure(figsize=(12, 6))
    plt.bar(range(1, k + 1), metrics["mse"], label="MSE")
    plt.bar(range(1, k + 1), metrics["mae"], label="MAE", alpha=0.7)
    plt.xlabel("Fold")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{label_name} - Evaluation Metrics per Fold")
    plt.savefig(os.path.join(save_metrics_dir, f"{label_name}_metrics_per_fold.png"))
    plt.close()

    return avg_metrics

# データ準備
bert_model_name = "cl-tohoku/bert-base-japanese"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name).to(device)

sentences, scores = [], []
with open(file_path, "r", encoding="utf-8") as file:
    for line in file.readlines()[1:]:
        parts = line.strip().split(",")
        if len(parts) == 2:
            sentences.append(parts[0].strip())
            scores.append(float(parts[1].strip()))

X = get_bert_embeddings(sentences, tokenizer, bert_model)
y = np.array(scores)

# 交差検証
cross_val_train_and_evaluate(X, y, "Dajudge", verbose=True)
