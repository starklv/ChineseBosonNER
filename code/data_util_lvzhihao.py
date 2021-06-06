#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import re
import pandas as pd
import collections
import numpy as np
import  torch.utils.data as Data
import torch

def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def data2pt():
    datas = []
    labels = []
    tags = set()
    with open("./wordtagsplit.txt", "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line = line.split()
            linedata = []
            linelabel = []
            numNot0 = 0
            for word in line:
                word = word.split('/')
                linedata.append(word[0])
                linelabel.append(word[1])
                tags.add(word[1])
                if word[1] != 'O':
                    numNot0 += 1
            if numNot0 != 0:
                datas.append(linedata)
                labels.append(linelabel)

    assert len(datas) == len(labels)

    all_words = flatten(datas)
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = range(1, len(set_words) + 1)

    tags = list(tags)
    tag_ids = range(len(tags))
    word2id = {}
    for idx, word in enumerate(set_words, start=1):
        word2id[word] = idx
    id2word = {value: key for key, value in word2id.items()}
    tag2id = {}
    for idx, tag in enumerate(tags):
        tag2id[tag] = idx
    id2tag = {value: key for key, value in tag2id.items()}

    def X_padding(words):
        ids = list(word2id[word] for word in words)
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    def y_padding(tags):
        ids = list(tag2id[tag] for tag in tags)
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
    df_data['x'] = df_data['words'].apply(X_padding)
    df_data['y'] = df_data['tags'].apply(y_padding)
    x = np.asarray(list(df_data['x'].values))
    y = np.asarray(list(df_data['y'].values))

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=43)

    def save_dataloader(input_x, input_y, name):
        dataset = Data.TensorDataset(torch.LongTensor(input_x), torch.tensor(input_y))
        dataloader = Data.DataLoader(dataset, batch_size=16, shuffle=True)
        torch.save(dataloader, f"{name}.pt")

    save_dataloader(x_train, y_train, "train")
    save_dataloader(x_test, y_test, "test")
    save_dataloader(x_valid, y_valid, "val")




def origin2tag(file_in_name, file_out_name):
    # input：{productname:浙江在线杭州}{time:4 月 25 日}讯（记者{personname: 施宇翔} 通讯员
    # output：浙/Bproductname 江/Mproductname 在/Mproductname 线/Mproductname
    # 就是把原始数据按照 BMEO 规则变成字标注的形式，以便模型训练；
    with open(file_in_name, 'r', encoding = 'utf-8') as fr, \
        open(file_out_name, "w", encoding = "utf-8") as fw:
        for line in fr.readlines():
            line = line.strip()
            i = 0
            while i < len(line):
                if line[i] == '{':
                    i += 2
                    temp = ""
                    while line[i] != '}':
                        temp += line[i]
                        i += 1
                    i += 2
                    word = temp.split(':')
                    sen = word[1]
                    fw.write(sen[0] + "/B_" + word[0] + " ")
                    for j in sen[1:len(sen) - 1]:
                        fw.write(j + "/M_" + word[0] + " ")
                    fw.write(sen[-1] + "/E_" +word[0] + " ")
                else:
                    fw.write(line[i] + "/O" + " ")
                    i += 1
            fw.write("\n")


def tagsplit(file_in_name, file_out_name):
    with open(file_in_name,"r", encoding = "utf-8") as fr, \
        open(file_out_name, "w", encoding = "utf-8") as fw:
        texts = fr.read()
        sentences = re.split('[，。！？、‘’“”（）]/[O]', texts)
        for sentence in sentences:
            if sentence != " ":
                fw.write(sentence.strip() + "\n")


origin2tag("./origindata.txt", tag_file)
tag_file = "./wordtag.txt"
tagsplit_file = "./wordtagsplit.txt"
tagsplit(tag_file, tagsplit_file)
