import jieba
import os
import re
import numpy as np


def getCorpus(text):
    r1 = u'[a-zA-Z0-9’"#$%&\'()（）*+-./:：;<=>@，★、…【】《》“”‘’[\\]^_`{|}~]+'
    text = re.sub(r1, "", text)
    p1 =["\t", "\n", "\u3000", "\u0020", "\u00A0", " "]
    for i in p1:
        text = text.replace(i, "")
    p2 = ["?", "？", "！", "!", ","]
    for i in p2:
        text = text.replace(i, "。")
    corpus = text.split("。")
    return corpus


def getWord(corpus, n=1):
    word_dict = {}
    for line in corpus:
        words = list(jieba.cut(line))
        for i in range(len(words) + 1 - n):
            key = tuple(words[i:i + n])
            word_dict[key] = word_dict.get(key, 0) + 1
    return word_dict


def calWordEntropy(corpus, n=1):
    if n > 1:
        word_dict_n = getWord(corpus, n)
        word_dict_n1 = getWord(corpus, n - 1)
        all_sum_n = np.sum(list(word_dict_n.values()))
        all_sum_n1 = np.sum(list(word_dict_n1.values()))
        entropy = -np.sum([k * np.log2(k / word_dict_n1[j[:n - 1]] * all_sum_n1 / all_sum_n) for j, k in word_dict_n.items()]) / all_sum_n
    else:
        word_dict = getWord(corpus)
        all_sum = np.sum(list(word_dict.values()))
        entropy = -np.sum([i * np.log2(i / all_sum) for i in word_dict.values()]) / all_sum
    return entropy


if __name__ == '__main__':
    path = "E:\桌面文件\课程学习类\研究生课程\研一下\DL-NLP\第一次大作业\jyxstxtqj_downcc.com"
    filedir = os.listdir(path)
    for text_file in filedir:
        text_position = path + '//' + text_file
        print(text_file)
        with open(text_position, 'r', encoding='ANSI') as f:
            data = f.read()
        f.close()
        corpus_file = getCorpus(data)

        cf_dict_word = getWord(corpus_file)
        word_num = np.sum(list(cf_dict_word.values()))
        print(str(word_num))

        for i in range(1, 4):
            entropy = calWordEntropy(corpus_file, i)
            print(f"{entropy:.4f}")
