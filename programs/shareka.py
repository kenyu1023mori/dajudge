import MeCab
from collections import Counter

class Shareka:
    def __init__(self, sentence, n=3):
        """置き換える文字リストが格納されたクラス変数"""
        self.replace_words = [["。", ""], ["、", ""], [",", ""], [".", ""], ["!", ""],
                              ["！", ""], ["・", ""], ["「", ""], ["」", ""], ["｣", ""],
                              ["『", ""], ["』", ""], [" ", ""], ["　", ""],
                              ["ッ", ""], ["ャ", "ヤ"], ["ュ", "ユ"], ["ョ", "ヨ"],
                              ["ァ", "ア"], ["ィ", "イ"], ["ゥ", "ウ"], ["ェ", "エ"], ["ォ", "オ"], ["ー", ""]]
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


# 入力と結果出力部分
if __name__ == "__main__":
    sentence = input("判定する文章を入力してください: ")
    
    shareka_instance = Shareka(sentence, n=3)
    
    print("\n===== 結果 =====")
    print(f"入力文（カタカナ変換）: {shareka_instance.kana}")
    
    if shareka_instance.dajarewake():
        print("判定: 駄洒落です。")
    else:
        print("判定: 駄洒落ではありません。")
