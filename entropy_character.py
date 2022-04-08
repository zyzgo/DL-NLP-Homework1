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


def getCharacter(corpus, n=1):
    char_dict = {}
    for line in corpus:
        for i in range(len(line)+1-n):
            key = "".join(line[i:i+n])
            char_dict[key] = char_dict.get(key, 0) + 1
    return char_dict


def calCharacterEntropy(corpus, n=1):
    if n > 1:
        char_dict_n = getCharacter(corpus, n)
        char_dict_n1 = getCharacter(corpus, n - 1)
        all_sum_n = np.sum(list(char_dict_n.values()))
        all_sum_n1 = np.sum(list(char_dict_n1.values()))
        entropy = -np.sum([k / all_sum_n * np.log2(k / char_dict_n1[j[:n - 1]] * all_sum_n1 / all_sum_n) for j, k in char_dict_n.items()])
    else:
        char_dict = getCharacter(corpus)
        all_sum = np.sum(list(char_dict.values()))
        entropy = -np.sum([i / all_sum * np.log2(i / all_sum) for i in char_dict.values()])
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

        cf_dict_char = getCharacter(corpus_file)
        char_num = np.sum(list(cf_dict_char.values()))
        print(str(char_num))

        for i in range(1, 4):
            entropy = calCharacterEntropy(corpus_file, i)
            print(f"{entropy:.4f}")
