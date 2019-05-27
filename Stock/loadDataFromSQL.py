# -*- coding: utf-8 -*-
from entity_mark import EntityMark
import numpy as np

def load_data_from_mysql(ratio = 0.8, dir="Stock_data"):
    entity_list = EntityMark.query.all()
    entity_num = len(entity_list)
    print("数据读取完毕！总计有%d行数据"%entity_num)
    content_list = []
    for entity in entity_list:
        content_list.append(entity.content)

    content_list = np.array(content_list)
    seed = 1
    np.random.seed(seed)
    np.random.shuffle(content_list)

    train_size = int(entity_num * ratio)
    #data2file(content_list[:train_size], dir, "train.txt")
    #data2file(content_list[train_size:], dir, "test.txt")
    getWord2Id(dir, content_list)
    print("文件存储完毕！")

def getWord2Id(dir, content_list, unk='<UNK>'):
    word_dict = {}
    for line in content_list:
        for word in line:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
    word_dict[unk] = len(word_dict)
    with open("%s/word2id.pkl" % dir, "wb") as word2id_file:
        import pickle
        pickle.dump(word_dict, word2id_file)

def data2file(data, dir, file_name):
    dir = dir + "/" if dir[-1] != '/' else dir
    with open(dir + file_name, "w") as file:
        for line in data:
            line = line + '\n' if line[-1] != '\n' else line
            file.write(line)

if __name__ == "__main__":
    load_data_from_mysql()
