#-*-encoding:utf8-*-#
"""
数据处理，包括：
1、原始数据（含某种分隔符）转逐字标注输入数据
2、输入数据做统计，根据指定的embedding文件生成char2id文件(使能控制oov是否产生unk)
 """
import re
from consoleLogger import logger
MSRA_data = {
    "pattern": "/o|/n[srt]",
    "tag": {"/o": "O", "/nr": "PER", "/ns": "LOC", "/nt": "ORG"},
    "other": "/o"
}


def original2inputFile(original_path, input_path, data_schema=MSRA_data, tag_schema="BIO"):
    delimiter, labels, other_tag = data_schema["pattern"], data_schema["tag"], data_schema["other"]
    with open(original_path, "r") as original_file:
        with open(input_path, "w") as input_file:
            line_cnt = 0
            suc_sen_cnt = 0
            entity_cnt = dict(zip(labels.keys(), [0]*len(labels)))
            for line in original_file:
                line_cnt += 1
                line = line.strip().replace(" ", "").replace("，", ",")
                if not line:
                    continue
                # 正则解析，依照预定义的标签拆解句子，获得一份文本列表和对应的标签列表
                tag_list = re.findall(delimiter, line)
                word_list = re.split(delimiter, line)
                if len(tag_list) < len(word_list) - 1:
                    logger.error("第%d行的标注数据解析异常，原因：标签数与拆解后的文本段数不和，文本内容：%s"%(line_cnt, line))
                if len(word_list[-1]):  # 有些人标的数据少了结尾的最后一处，补上个o
                    tag_list.append(other_tag)  # 在去掉了空格的正常情况下，这里的最后一个应该是一个空字符串，不是的话说明没有标注到句尾
                char_list, char_tag_list, entity_cnt = sent2char(word_list, tag_list, other_tag, entity_cnt, labels, tag_schema)
                if char_list and char_tag_list:
                    suc_sen_cnt += 1
                    # 一句话的结果写入文件
                    for char, tag in zip(char_list, char_tag_list):
                        input_file.write('%s %s\n' % (char, tag))
                    input_file.write('\n')  # 一句话结束后一个空行
            logger.info("处理完毕，数据文件总行数%d，实际插入成功行数%d，各类实体的数据情况如下：%s"
                        % (line_cnt, suc_sen_cnt, str(entity_cnt)))


def sent2char(word_list, tag_list, other_tag, entity_cnt, labels, tag_schema):
    char_list = []
    char_tag_list = []
    for i in range(len(tag_list)):
        tag = tag_list[i]
        entity_cnt[tag] += 1  # 实体计数+1
        word = word_list[i]
        for c in range(len(word)):
            char_list.append(word[c])
            prefix = ""
            if tag != other_tag:  # 不是Other类，则区分BIO编码和BIEO
                if tag_schema == "BIO":
                    prefix = "B-" if not c else "I-"
                elif tag_schema == "BIEO":
                    prefix = "B-" if not c else "E-" if c == len(word) - 1 else "I-"
            char_tag_list.append(prefix + labels[tag])
    return char_list, char_tag_list, entity_cnt

if __name__=="__main__":
    MSRA_original = "data_path/original/testright1.txt"
    MSRA_input = "data_path/MSRA/input/test_data"
    original2inputFile(original_path=MSRA_original, input_path=MSRA_input)
    # MSRA训练集：各类实体的数据情况如下：{'/nr': 17615, '/ns': 36517, '/nt': 20571, '/o': 1193462}
    # 各类实体的数据情况如下：{'/nr': 1973, '/ns': 2877, '/o': 8786, '/nt': 1331}
    # cat train_data | grep -n ^.$ > record
