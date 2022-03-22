#!/usr/bin/env python
# -*- coding:utf-8 _*-
""" 
@author:quincy qiang
@license: Apache Licence
@file: 01_crf.py
@time: 2022/03/22
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""

import re
import warnings

import joblib
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics
from tqdm import tqdm

warnings.filterwarnings('ignore')


## 加载数据
def load_data(data_path):
    data = list()
    data_sent_with_label = list()
    with open(data_path, mode='r', encoding="utf-8") as f:
        for line in tqdm(f):
            if line.strip() == "":
                data.append(data_sent_with_label.copy())
                data_sent_with_label.clear()
            else:
                row_data = line.strip().split(" ")
                if len(row_data) == 1:
                    data_sent_with_label.append((' ', row_data[0]))
                else:
                    data_sent_with_label.append(tuple(line.strip().split(" ")))
    return data


## 构造ngram特征
def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word': word,
        'word.isdigit()': word.isdigit(),
        'word.isspace()': word.isspace(),
        'word.isalpha()': word.isalpha(),

    }
    if i > 0:
        word1 = sent[i - 1][0]
        words = word1 + word
        features.update({
            '-1:word': word1,
            '-1:words': words,
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.isspace()': word1.isalpha(),

        })
    else:
        features['BOS'] = True

    #     if i > 1:
    #         word2 = sent[i-2][0]
    #         word1 = sent[i-1][0]
    #         words = word1 + word2 + word
    #         features.update({
    #             '-2:word': word2,
    #             '-2:words': words,
    #             '-2:word.isdigit()': word2.isdigit(),
    #             '-2:word.isspace()': word2.isalpha(),

    #         })

    # if i > 2:
    #     word3 = sent[i - 3][0]
    #     word2 = sent[i - 2][0]
    #     word1 = sent[i - 1][0]
    #     words = word1 + word2 + word3 + word
    #     features.update({
    #         '-3:word': word3,
    #         '-3:words': words,
    #         '-3:word.isdigit()': word3.isdigit(),
    #         '-3:word.isspace()': word3.isalpha(),
    #     })

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        words = word1 + word
        features.update({
            '+1:word': word1,
            '+1:words': words,
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.isspace()': word1.isalpha(),
        })
    else:
        features['EOS'] = True

    #     if i < len(sent)-2:
    #         word2 = sent[i + 2][0]
    #         word1 = sent[i + 1][0]
    #         words = word + word1 + word2
    #         features.update({
    #             '+2:word': word2,
    #             '+2:words': words,
    #             '+2:word.isdigit()': word2.isdigit(),
    #         })

    # if i < len(sent)-3:
    #     word3 = sent[i + 3][0]
    #     word2 = sent[i + 2][0]
    #     word1 = sent[i + 1][0]
    #     words = word + word1 + word2 + word3
    #     features.update({
    #         '+3:word': word3,
    #         '+3:words': words,
    #         '+3:word.isdigit()': word3.isdigit(),
    #     })

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [ele[-1] for ele in sent]


def train_crf():
    train = load_data('data/train_data/train.txt')

    train, valid = train_test_split(train, test_size=0.2, shuffle=True, random_state=42)
    print(len(train), len(valid))

    # 生成特征
    X_train = [sent2features(s) for s in tqdm(train)]
    y_train = [sent2labels(s) for s in tqdm(train)]

    X_dev = [sent2features(s) for s in tqdm(valid)]
    y_dev = [sent2labels(s) for s in tqdm(valid)]

    # **表示该位置接受任意多个关键字（keyword）参数，在函数**位置上转化为词典 [key:value, key:value ]
    crf_model = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.25, c2=0.018, max_iterations=100,
                                     all_possible_transitions=True, verbose=True)
    crf_model.fit(X_train, y_train)

    labels = list(crf_model.classes_)
    labels.remove("O")
    y_pred = crf_model.predict(X_dev)
    metrics.flat_f1_score(y_dev, y_pred,
                          average='weighted', labels=labels)
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    print(metrics.flat_classification_report(
        y_dev, y_pred, labels=sorted_labels, digits=3
    ))

    joblib.dump(crf_model, "./product_crf_model.joblib")


def predict():
    test_file = 'data/preliminary_test_a/sample_per_line_preliminary_A.txt'
    test_sents = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f.read().split('\n'):
            test_sents.append(line)

    sents_feature = [sent2features(sent) for sent in test_sents]
    # text = 'OPPO闪充充电器 X9070 X9077 R5 快充头通用手机数据线 套餐【2.4充电头+数据线 】 安卓 1.5m'
    NER_tagger = joblib.load('./product_crf_model.joblib')
    list_results = []

    y_pred = NER_tagger.predict(sents_feature)

    for sent, ner_tag in zip(test_sents, y_pred):
        line_result = []
        for word, tag in zip(sent, ner_tag):
            line_result.append((word, tag))
        list_results.append(line_result)
    with open('crf.txt', 'w', encoding='utf-8') as f:
        for i, line_result in enumerate(list_results):
            for word, tag in line_result:
                f.write(f'{word} {tag}\n')
            if i < len(list_results) - 1:
                f.write('\n')


if __name__ == '__main__':
    # train_crf()

    predict()
