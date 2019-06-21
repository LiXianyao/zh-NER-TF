#-*-encoding:utf8-*-#
from itertools import combinations
import sys
sys.path.append("..")
import time
from consoleLogger import logger
"""
将测试数据/模型输出数据，转化为关系抽取的输入数据，格式为：
csv文件：一列
    文本标注
    句子<e1>实体1</e1>..<e2>实体1</e2>...
"""


def current_timestamp():
    timestamp = time.strftime("%m%d%H%M", time.localtime())
    return timestamp

class RelationDataTransformer:
    init_suffix = "O"  # 标签后缀的初值

    def __init__(self, data_path, save_dir="relationData/{}", file_name=current_timestamp()):
        result_name = save_dir.format(file_name)
        with open(data_path, "r", encoding="utf-8") as entity_data:
            with open(result_name, "w", encoding="utf-8") as result_file:
                self.result_cnt = 0
                self.result_file = result_file
                logger.info("开始转换数据，源文件名：{}.....".format(data_path))
                self.result2Relation(entity_data)

    def init_sentence(self):  # 初始化一句话需要记录的所有信息
        entity_set = set()  # 一句话的所有实体的头尾位置tuple集合， element = (head idx, tail idx)
        sentence = ""  # 一句话的所有字符
        suffix = self.init_suffix  # 标签后缀的初值
        head = tail = 0
        return entity_set, sentence, suffix, head, tail

    def result2Relation(self, entity_data):
        """
        :param entity_data: 每行空格隔开的两列  字 字的label
        :return:
        首先对数据进行过滤，确定句子边界，对每一句数据，确定其中包含的所有实体(的起止位置)，然后做两两组合
        类尺取法，只要标签（后缀）发生了变化，就更新记录一个头尾指针， 因为O没有后缀还是O，很容易判定是否计入实体集合
        """
        entity_set, sentence, pre_suffix, head, tail = self.init_sentence()
        sen_cnt = 0
        entity_cnt = 0
        for data in entity_data:
            if data != '\n':
                [char, label] = data.strip().split()
                suffix = label.split("-")[-1]
                sentence += char
                if suffix != pre_suffix:  # 后缀发生了变化，即label的实体类型发生了变化，处理记录
                    if pre_suffix != "O":  # 前面的一段head~tail对应一个非O实体
                        entity_set.add((head, tail))
                    head = tail  #
                tail = tail + 1
                pre_suffix = suffix  # 移动处理下一个字

            else:  # 遇到了一句话的结尾
                if len(sentence.strip()):  # 句子不为空，处理并输出到文件
                    if pre_suffix != "O":  # 句子最后的head~tail对应一个非O实体，但没有被处理到
                        entity_set.add((head, tail))
                    entity_cnt += len(entity_set)
                    sen_cnt += 1
                    self.save_relation_data(sentence, entity_set)
                    if sen_cnt == 10:
                        break
                entity_set, sentence, pre_suffix, head, tail = self.init_sentence()
        logger.info("数据转换完毕，总计有{}条实体数据，共{}个实体，产生了{}条关系数据".format(sen_cnt, entity_cnt, self.result_cnt))

    def save_relation_data(self, sentence, entity_set):
        loc2enti = self.trans_entity
        #print(sentence)
        entity_comb = [combi for combi in combinations(entity_set, 2)]
        #print("总计有%d个实体，构成%d个不同的实体对" % (len(entity_set), len(entity_comb)))
        #print(str([loc2enti(sentence, entity) for entity in entity_set]))
        for combi in entity_comb:
            relation_sample = self.trans_relation_data(sentence, sorted(combi))  # 每个实体对生成一个样本数据
            #print(relation_sample)
            self.result_file.write(relation_sample)  # 关系数据写入文件
            self.result_file.write("\n")
            self.result_cnt += 1


    def trans_entity(self, sentence, entity_location):
        (entity_head, entity_tail) = entity_location
        entity = sentence[entity_head: entity_tail]
        return entity

    def trans_relation_data(self, sentence, entity_loc_pair):
        entity_tag = lambda id, en_str: "<e{}>{}</e{}>".format(id + 1, en_str, id + 1)
        result_str = ""
        last_loc = 0
        for idx in range(2):
            (entity_head, entity_tail) = entity_loc_pair[idx]
            entity = sentence[entity_head: entity_tail]
            result_str += sentence[last_loc: entity_head]\
                          + entity_tag(idx, entity)  # 实体前面的内容（到上一个实体为止）+ 当前实体
            last_loc = entity_tail
        result_str += sentence[last_loc:]  # 补上第二个实体后面的内容
        return result_str

if __name__=="__main__":
    data_path = "../Stock/Stock_data/test_data_c"
    relationTransformer = RelationDataTransformer(data_path)
