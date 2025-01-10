import MeCab
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

# ダジャレかどうか判別するAI
# 以下の記事を参照
# https://qiita.com/fujit33/items/dbfbd7a2aa3858067b6c#shareka%E3%83%80%E3%82%B8%E3%83%A3%E3%83%AC%E5%88%A4%E5%88%A5ai
'''
ダジャレデータ1000個
非ダジャレデータ1000個
n=3
===== 精度評価 =====
正解率 (Accuracy): 0.89
適合率 (Precision): 0.94
再現率 (Recall): 0.83
F1スコア: 0.88
'''
class Shareka:
    def __init__(self, sentence, n=2):
        """置き換える文字リストが格納されたクラス変数"""
        self.replace_words = [["。", ""], ["、", ""], [",", ""], [".", ""], ["!", ""],
                              ["！", ""], ["・", ""], ["「", ""], ["」", ""], ["｣", ""],
                              ["『", ""], ["』", ""], [" ", ""], ["　", ""],
                              ["ッ", ""], ["ャ", "ヤ"], ["ュ", "ユ"], ["ョ", "ヨ"],
                              ["ァ", "ア"], ["ィ", "イ"], ["ゥ", "ウ"], ["ェ", "エ"], ["ォ", "オ"], ["ー", ""], ["ヲ", "オ"]]
        self.kaburi = n
        self.sentence = sentence

        mecab = MeCab.Tagger("-Oyomi")
        self.kana = mecab.parse(sentence).strip()  # 改行を削除
        self.preprocessed = self.preprocessing(self.kana)
        self.devided = self.devide(self.preprocessed)

    def preprocessing(self, sentence):
        """文字列を置き換える前処理"""
        for replace_word in self.replace_words:
            sentence = sentence.replace(replace_word[0], replace_word[1])
        return sentence

    def devide(self, sentence):
        """文字列を指定された長さで分割"""
        elements = []
        repeat_num = len(sentence) - (self.kaburi - 1)
        for i in range(repeat_num):
            elements.append(sentence[i:i+self.kaburi])
        return elements

    def has_duplicates(self):
        """重複する部分文字列があるかどうかを判定"""
        if not self.devided:
            return False
        counter = Counter(self.devided)
        for value in counter.values():
            if value > 1:
                return True
        return False

    def dajarewake(self):
        """駄洒落判定"""
        return self.has_duplicates()

# CSVを読み込み、判定と精度評価
if __name__ == "__main__":
    # CSVファイルのパス
    csv_file = "../../data/shareka_test.csv"

    # データの読み込み
    data = pd.read_csv(csv_file, header=0)

    # ラベルを整数型に変換
    data['label'] = data['label'].astype(int)

    # 駄洒落判定結果を格納するリスト
    predictions = []

    for sentence in data["sentence"]:
        shareka_instance = Shareka(sentence, n=3)
        predictions.append(int(shareka_instance.dajarewake()))

    # 正解ラベル
    true_labels = data["label"].tolist()

    # 精度評価
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    print("===== 精度評価 =====")
    print(f"正解率 (Accuracy): {accuracy:.2f}")
    print(f"適合率 (Precision): {precision:.2f}")
    print(f"再現率 (Recall): {recall:.2f}")
    print(f"F1スコア: {f1:.2f}")
