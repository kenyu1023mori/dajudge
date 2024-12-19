import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import KFold
import numpy as np
from transformers import BertJapaneseTokenizer, BertModel  # Update import for tokenizer
import matplotlib.pyplot as plt  # matplotlibをインポート

# データパスおよび保存先ディレクトリ設定
dataset_path = "../../data/filtered_dajare_2.csv"
version = "v1.10"
save_model_dir = f"../models/{version}"
os.makedirs(save_model_dir, exist_ok=True)
save_metrics_dir = f"../metrics/{version}"
os.makedirs(save_metrics_dir, exist_ok=True)

# GPU/CPUの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# BERTトークナイザとモデル
pretrained_model_name = "cl-tohoku/bert-base-japanese"
tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model_name)  # Update tokenizer class
bert_model = BertModel.from_pretrained(pretrained_model_name).to(device)

# Gradient Checkpointing 有効化
bert_model.gradient_checkpointing_enable()

# ハイパーパラメータの定義
hyperparameters = {
    "learning_rate": 0.01,  # 学習率
    "num_epochs": 5,  # エポック数
    "dropout_rate": 0.3,  # ドロップアウト率
    "max_length": 64,  # トークンの最大長
    "batch_size": 16,  # バッチサイズ
    "k_folds": 5  # 交差検証の分割数
}

# データセットの定義
class DajareDataset(Dataset):
    def __init__(self, sentences, scores, tokenizer, max_length=64):
        self.sentences = sentences
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        score = self.scores[idx]
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return input_ids, attention_mask, torch.tensor(score, dtype=torch.float32)

# ニューラルネットワークモデル
class DajarePredictor(nn.Module):
    def __init__(self, bert_model):
        super(DajarePredictor, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS]トークンの埋め込み
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x

# データの読み込み
sentences, scores = [], []
with open(dataset_path, "r", encoding="utf-8") as file:
    next(file)  # ヘッダー行をスキップ
    for line in file:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            sentences.append(parts[0].strip())
            scores.append(float(parts[1].strip()))

# データサイズの確認
print(f"Total sentences: {len(sentences)}, Total scores: {len(scores)}")

# データの一部を表示して確認
print("Sample sentences:", sentences[:5])
print("Sample scores:", scores[:5])

# 交差検証とモデルの訓練・評価
def cross_val_train_and_evaluate(sentences, scores, tokenizer, bert_model, label_name, hyperparameters):
    k_folds = hyperparameters["k_folds"]
    batch_size = hyperparameters["batch_size"]
    learning_rate = hyperparameters["learning_rate"]
    num_epochs = hyperparameters["num_epochs"]
    dropout_rate = hyperparameters["dropout_rate"]
    max_length = hyperparameters["max_length"]

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(sentences)):
        print(f"Fold {fold + 1}/{k_folds} 開始")
        train_sentences = [sentences[i] for i in train_idx]
        val_sentences = [sentences[i] for i in val_idx]
        train_scores = [scores[i] for i in train_idx]
        val_scores = [scores[i] for i in val_idx]

        train_dataset = DajareDataset(train_sentences, train_scores, tokenizer, max_length=max_length)
        val_dataset = DajareDataset(val_sentences, val_scores, tokenizer, max_length=max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = DajarePredictor(bert_model).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Mixed Precision Training のスケーラー
        scaler = torch.amp.GradScaler()

        # モデルの訓練
        for epoch in range(num_epochs):  # エポック数を調整可能
            print(f"Epoch {epoch + 1}/{num_epochs} 開始")
            model.train()
            for batch_idx, (input_ids, attention_mask, scores) in enumerate(train_loader):
                input_ids, attention_mask, scores = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    scores.to(device).view(-1, 1),
                )
                optimizer.zero_grad()

                # Mixed Precision Training
                with torch.amp.autocast(device_type='cuda'):
                    predictions = model(input_ids, attention_mask)
                    loss = criterion(predictions, scores)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # 進行状況を適度に表示
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

        # エポック終了時にモデルを保存
        torch.save(model.state_dict(), os.path.join(save_model_dir, f"{label_name}_fold_{fold + 1}.pth"))
        print(f"モデル保存: {label_name}_fold_{fold + 1}.pth")

        # 検証
        model.eval()
        val_mse_loss = 0.0
        val_mae_score = 0.0
        with torch.no_grad():
            for input_ids, attention_mask, scores in val_loader:
                input_ids, attention_mask, scores = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    scores.to(device).view(-1, 1),
                )
                predictions = model(input_ids, attention_mask)
                mse_loss = criterion(predictions, scores).item()
                mae_score = torch.mean(torch.abs(predictions - scores)).item()
                val_mse_loss += mse_loss
                val_mae_score += mae_score

        val_mse_loss /= len(val_loader)
        val_mae_score /= len(val_loader)
        print(f"{label_name} - Fold {fold + 1}/{k_folds} - Validation MSE Loss: {val_mse_loss}, Validation MAE: {val_mae_score}")

        # 結果保存
        metrics = {
            "Validation MSE": val_mse_loss,
            "Validation MAE": val_mae_score
        }
        fold_metrics.append(metrics)

        with open(os.path.join(save_metrics_dir, f"{label_name}_fold_{fold + 1}_metrics.pkl"), "wb") as f:
            pickle.dump(metrics, f)
        print(f"{label_name} - Fold {fold + 1} - Metrics saved")

    # 平均結果を保存
    avg_mse = np.mean([m["Validation MSE"] for m in fold_metrics])
    avg_mae = np.mean([m["Validation MAE"] for m in fold_metrics])
    metrics = {"Average Validation MSE": avg_mse, "Average Validation MAE": avg_mae}

    # 棒グラフの作成と保存
    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['blue', 'blue'])
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title(f'{label_name} - Average Metrics')
    plt.savefig(os.path.join(save_metrics_dir, f"{label_name}_average_metrics.png"))
    plt.close()

    print(f"{label_name} - Average Metrics saved as image")

# 実行
cross_val_train_and_evaluate(sentences, scores, tokenizer, bert_model, "Dajudge", hyperparameters)
