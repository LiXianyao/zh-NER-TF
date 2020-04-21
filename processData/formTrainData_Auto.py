#-*-encoding:utf8-*-#
"""
数据处理，包括：
1、原始数据（含某种分隔符）转逐字标注输入数据
2、输入数据做统计，根据指定的embedding文件生成char2id文件(使能控制oov是否产生unk)
3、由于现在是请专人标注的数据，且没有打字错误，考虑去掉一些原有的逻辑
 """
import re
import sys
sys.path.append("..")
from consoleLogger import logger
MSRA_data = {
    "pattern": "/o|/n[srt]",
    "tag": {"/o": "O", "/nr": "PER", "/ns": "LOC", "/nt": "ORG"},
    "other": "/o"
}

Stock_data = {
    "pattern": "</o>|</n[oifnptmlc]>",
    "tag": {"</o>": "O", "</ni>": "PROD", "</nf>": "FIE", "</no>": "ORG",
            "</nn>": "NUM", "</np>": "PERC", "</nt>": "TIME", "</nm>": "MERT",
            "</nl>": "LOC", "</nc>": "ISO"},
    "other": ["</o>"]  # ["</o>", "</nl>", "</nn>", "</nm>", "</nc>"]
}


def original2inputFile(original_path, input_path, data_schema=MSRA_data, tag_schema="BIO"):
    delimiter, labels, other_tag = data_schema["pattern"], data_schema["tag"], data_schema["other"]
    #entity_dict = get_entity_dict(original_path, data_schema)

    with open(original_path, "r") as original_file:
        with open(input_path, "w", encoding="utf8") as input_file:
            line_cnt = 0
            suc_sen_cnt = 0
            entity_cnt = dict(zip(labels.keys(), [0]*len(labels)))
            for line in original_file:
                line_cnt += 1
                line = line.strip().replace(" ", "").replace("，", ",")
                #line = content_completion(line)  ## 进行补全
                if not line:
                    continue
                tag_list, word_list = line2seq(line, line_cnt, delimiter, other_tag)
                char_list, char_tag_list, entity_cnt = sent2char(word_list, tag_list, other_tag, entity_cnt, labels, tag_schema)
                if char_list and char_tag_list:
                    suc_sen_cnt += 1
                    # 一句话的结果写入文件
                    for char, tag in zip(char_list, char_tag_list):
                        input_file.write('%s %s\n' % (char, tag))
                    input_file.write('\n')  # 一句话结束后一个空行
            """ 对所有label生成对应的映射文件 """
            #store_label_file(labels, other_tag, tag_schema, input_path)

            """ 计算输出每种label在总label重的占比（O除外）"""
            entity_sum = 0
            for entity in entity_cnt:
                if entity != other_tag:
                    entity_sum += entity_cnt[entity]
            for entity in entity_cnt:
                if entity not in other_tag:
                    percentage = entity_cnt[entity] * 1. / entity_sum
                    entity_cnt[entity] = (entity_cnt[entity], round(percentage, 2))
            logger.info("处理完毕，数据文件总行数%d，实际插入成功行数%d，各类实体的数据情况如下：%s"
                        % (line_cnt, suc_sen_cnt, str(sorted(entity_cnt.items()))))


def store_label_file(labels, other_tag, tag_schema, input_path):
    tag_schema = tag_schema[:-1]
    """ 对所有label生成对应的映射文件 """
    label_dict = {}
    label_dict[labels[other_tag[0]]] = 0 # 第一个必须是O
    for label in labels:
        if not label == other_tag[0]:
            for loc in tag_schema:
                loc_label = "%s-%s" % (loc, labels[label])
                label_dict[loc_label] = len(label_dict)
    input_dir = "/".join(input_path.split("/")[:-1])
    print("input_dir is %s"%input_dir)
    with open("%s/%s" %(input_dir, "labels_data"), "w") as label_file:
        import json
        json.dump(label_dict, label_file)

"""
根据输入的文本段落列表word_list及每个段落对应的标注列表tag_list
将一个段落里的所有字逐个打上对应的BIO（或BIEO）标注
"""
def sent2char(word_list, tag_list, other_tag, entity_cnt, labels, tag_schema):
    char_list = []
    char_tag_list = []
    for i in range(len(tag_list)):
        tag = tag_list[i]
        tag = other_tag[0] if tag in other_tag else tag
        word = word_list[i]
        #if tag != other_tag: # 是个标注实体，直接取dict里的
        #    tag = entity_dict[word] if word in entity_dict else tag

        entity_cnt[tag] += 1  # 实体计数+1
        for c in range(len(word)):
            char_list.append(word[c])
            prefix = ""
            if tag not in other_tag:  # 不是Other类，则区分BIO编码和BIEO
                if tag_schema == "BIO":
                    prefix = "B-" if not c else "I-"
                elif tag_schema == "BIEO":
                    prefix = "B-" if not c else "E-" if c == len(word) - 1 else "I-"
            char_tag_list.append(prefix + labels[tag])
    return char_list, char_tag_list, entity_cnt

def vec2id(vec_path, data_path, name="chn"):
    """ 处理pretrain的vec文件gigaword_chn.all.a2b.uni.ite50.vec，生成一个word2id.pkl和一个 gigaword_chn.npy"""
    with open(vec_path, "r") as vec_file:
        word_dict = {}
        embedding = []
        for line in vec_file:
            line = line.strip().split()
            assert len(line) > 2
            word = line[0]
            embedding.append(line[1:])
            word_dict[word] = len(word_dict)
        logger.info("数据读取处理完毕，总计有%d个字，每个字的embedding长度为%d"%(len(word_dict), len(embedding[0])))
        import numpy as np, pickle
        embedding = np.array(embedding)
        print(embedding.shape)
        np.save(data_path + name + ".npy", embedding) #保存embedding到文件
        logger.info("embedding文件保存完毕！")
        with open(data_path + name + ".pkl", "wb") as word2id_file:
            pickle.dump(word_dict, word2id_file)
        logger.info("word2id文件保存完毕！")

"""
将一行标注数据拆分成两个列表：文本段落 - 段落对应的实体类型
"""
def line2seq(line, line_cnt, delimiter, other_tag):
    # 正则解析，依照预定义的标签拆解句子，获得一份文本列表和对应的标签列表
    tag_list = re.findall(delimiter, line)
    word_list = re.split(delimiter, line)
    if len(tag_list) < len(word_list) - 1:
        logger.error("第%d行的标注数据解析异常，原因：标签数与拆解后的文本段数不和，文本内容：%s" % (line_cnt, line))
    if len(word_list[-1]):  # 有些人标的数据少了结尾的最后一处，补上个o
        tag_list.append(other_tag[0])  # 在去掉了空格的正常情况下，这里的最后一个应该是一个空字符串，不是的话说明没有标注到句尾
    if not len(word_list[0].strip()):  # 句首多了个标签
        del tag_list[0]
        del word_list[0]
    return tag_list, word_list



if __name__=="__main__":
    #"""
    MSRA_original = "../data_path/original/testright1.txt"
    MSRA_input = "../data_path/MSRA/test_data"
    Stock_train_original = "../Stock/Stock_data/train_0930.txt"
    Stock_train_input = "../Stock/Stock_data/train_data_0930"

    original2inputFile(original_path=Stock_train_original, input_path=Stock_train_input, data_schema=Stock_data)
    Stock_test_original = "../Stock/Stock_data/test_0930.txt"
    Stock_test_input = "../Stock/Stock_data/test_data_0930"
    #entity_dict_pre = get_entity_dict(original_path=Stock_train_original, data_schema=Stock_data)
    original2inputFile(original_path=Stock_test_original, input_path=Stock_test_input, data_schema=Stock_data)

    #0921
    #[('</nc>', (6, 0.0)), ('</nf>', (729, 0.09)), ('</ni>', (2394, 0.29)), ('</nl>', (244, 0.03)), ('</nm>', (52, 0.01)),
    # ('</nn>', (328, 0.04)), ('</no>', (3185, 0.39)), ('</np>', (258, 0.03)), ('</nt>', (951, 0.12)), ('</o>', 8468)]
    #[('</nc>', (0, 0.0)), ('</nf>', (351, 0.1)), ('</ni>', (954, 0.29)), ('</nl>', (108, 0.03)), ('</nm>', (28, 0.01)),
    # ('</nn>', (128, 0.04)), ('</no>', (1306, 0.39)), ('</np>', (94, 0.03)), ('</nt>', (378, 0.11)), ('</o>', 3532)]

    #0930
    #[('</nc>', (6, 0.0)), ('</nf>', (1151, 0.05)), ('</ni>', (3596, 0.15)), ('</nl>', (341, 0.01)), ('</nm>', (108, 0.0)),
    # ('</nn>', (459, 0.02)), ('</no>', (4599, 0.19)), ('</np>', (405, 0.02)), ('</nt>', (1441, 0.06)), ('</o>', 12662)]
    #[('</nc>', (0, 0.0)), ('</nf>', (517, 0.05)), ('</ni>', (1594, 0.15)), ('</nl>', (188, 0.02)), ('</nm>', (42, 0.0)),
    # ('</nn>', (183, 0.02)), ('</no>', (1959, 0.18)), ('</np>', (161, 0.01)), ('</nt>', (653, 0.06)), ('</o>', 5504)]