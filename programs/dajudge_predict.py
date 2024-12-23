import sys
import os
import torch
import torch.nn as nn
from transformers import BertJapaneseTokenizer, BertModel
import MeCab

# 必要な変数とパスを設定
version = "v1.11"
load_dir = f"../models/{version}"
pretrained_model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# MeCabの設定
mecab = MeCab.Tagger("-Owakati")

# ニューラルネットワークモデルのクラス定義
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

# モデルのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model_name)
bert_model = BertModel.from_pretrained(pretrained_model_name)
model = DajarePredictor(bert_model)
model_path = os.path.join(load_dir, "Dajudge_epoch_5.pth")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 入力したダジャレに対してモデルのスコアを出力する関数
def predict_score(input_text, model, tokenizer, mecab):
    tokenized_text = mecab.parse(input_text).strip()
    encoding = tokenizer(
        tokenized_text,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        prediction = model(input_ids, attention_mask).squeeze().item()
        predicted_class = round(prediction)
        predicted_class = max(1, min(predicted_class, 5))
        print(f"Predicted Score: {predicted_class}")

# 標準入力のエンコーディングを明示的に設定
sys.stdin.reconfigure(encoding='utf-8')

# ユーザー入力処理
while True:
    try:
        input_text = input("Enter a Dajare (or type 'q' to quit): ").strip()
        if input_text.lower() == 'q':
            break
        predict_score(input_text, model, tokenizer, mecab)
    except UnicodeDecodeError as e:
        print(f"Encoding Error: {e}. Please try again.")
