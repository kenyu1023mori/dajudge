import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from transformers import BertJapaneseTokenizer, BertModel  # Update import for tokenizer

# データパスおよび保存先ディレクトリ設定
dataset_path = "../../data/filtered_dajare_2.csv"
version = "v1.11"
save_model_dir = f"../models/{version}"
os.makedirs(save_model_dir, exist_ok=True)
save_metrics_dir = f"../metrics/{version}"
os.makedirs(save_metrics_dir, exist_ok=True)

# GPU/CPUの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# BERTトークナイザとモデル
pretrained_model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model_name)  # Update tokenizer class
bert_model = BertModel.from_pretrained(pretrained_model_name).to(device)

# Gradient Checkpointing 有効化
bert_model.gradient_checkpointing_enable()

# ハイパーパラメータの定義
hyperparameters = {
    "learning_rate": 0.001,  # Adjusted learning rate
    "num_epochs": 5,  # エポック数
    "dropout_rate": 0.3,  # ドロップアウト率
    "max_length": 64,  # トークンの最大長
    "batch_size": 16,  # バッチサイズ
    "train_split": 0.8,  # 訓練データの割合
    "val_split": 0.1,  # 検証データの割合
    "test_split": 0.1  # テストデータの割合
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
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS]トークンの埋め込み
        x = self.dropout(pooled_output)
        x = self.fc(x)
        x = self.sigmoid(x)  # Apply sigmoid activation
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

# Normalize the scores to a range between 0 and 1
max_score = max(scores)
min_score = min(scores)
scores = [(score - min_score) / (max_score - min_score) for score in scores]

# データサイズの確認
print(f"Total sentences: {len(sentences)}, Total scores: {len(scores)}")

# データの一部を表示して確認
print("Sample sentences:", sentences[:5])
print("Sample scores:", scores[:5])

# データセットの分割
dataset = DajareDataset(sentences, scores, tokenizer, max_length=hyperparameters["max_length"])
train_size = int(hyperparameters["train_split"] * len(dataset))
val_size = int(hyperparameters["val_split"] * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)

# モデルの訓練と評価
def train_and_evaluate(train_loader, val_loader, test_loader, tokenizer, bert_model, label_name, hyperparameters):
    learning_rate = hyperparameters["learning_rate"]
    num_epochs = hyperparameters["num_epochs"]
    dropout_rate = hyperparameters["dropout_rate"]

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
        torch.save(model.state_dict(), os.path.join(save_model_dir, f"{label_name}_epoch_{epoch + 1}.pth"))
        print(f"モデル保存: {label_name}_epoch_{epoch + 1}.pth")

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
    print(f"{label_name} - Validation MSE Loss: {val_mse_loss}, Validation MAE: {val_mae_score}")

    # テスト
    test_mse_loss = 0.0
    test_mae_score = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, scores in test_loader:
            input_ids, attention_mask, scores = (
                input_ids.to(device),
                attention_mask.to(device),
                scores.to(device).view(-1, 1),
            )
            predictions = model(input_ids, attention_mask)
            mse_loss = criterion(predictions, scores).item()
            mae_score = torch.mean(torch.abs(predictions - scores)).item()
            test_mse_loss += mse_loss
            test_mae_score += mae_score

    test_mse_loss /= len(test_loader)
    test_mae_score /= len(test_loader)
    print(f"{label_name} - Test MSE Loss: {test_mse_loss}, Test MAE: {test_mae_score}")

    # 結果保存
    metrics = {
        "Validation MSE": val_mse_loss,
        "Validation MAE": val_mae_score,
        "Test MSE": test_mse_loss,
        "Test MAE": test_mae_score
    }
    with open(os.path.join(save_metrics_dir, f"{label_name}_metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    print(f"{label_name} - Metrics saved")

# 実行
train_and_evaluate(train_loader, val_loader, test_loader, tokenizer, bert_model, "Dajudge", hyperparameters)
