import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from itertools import combinations
import MeCab

# データの読み込みと前処理
limit = int(10e8)  # データ数の上限
sentences, scores = [], []
with open(file_path, "r", encoding="utf-8") as file:
    for line in file.readlines()[24:]:
        if len(sentences) >= limit:
            break
        parts = line.strip().split(",")
        if len(parts) >= 5:
            sentences.append(parts[1].strip())
            scores.append(float(parts[-1].strip()))

# MeCabを初期化して文を分かち書き
mecab = MeCab.Tagger("-Owakati")  # MeCabを分かち書きモードで使用

# 文を分割する関数
def tokenize_sentence(sentence):
    return mecab.parse(sentence).strip()

# 全文を分割（単語化）
tokenized_sentences = [tokenize_sentence(sentence).split() for sentence in sentences]

# Word2Vecモデルのトレーニング
w2v_model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
w2v_model.save("/home/group4/evaluate_dajare/models/word2vec_dajare.model")
