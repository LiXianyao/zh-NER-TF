#-*-encoding:utf8-*-#
"""
将测试数据/模型输出数据，转化为关系抽取的输入数据，格式为：
csv文件：一列
    文本标注
    句子<e1>实体1</e1>..<e2>实体1</e2>...
"""
class RelationDataTransformer:
    init_suffix = "O"  # 标签后缀的初值

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
        for data in entity_data:
            if data != '\n':
                [char, label] = data.strip().splite()
                suffix = label.split("-")[-1]
                sentence += char
                if suffix != pre_suffix:  # 后缀发生了变化，即label的实体类型发生了变化，处理记录
                    if pre_suffix != "O":  # 前面的一段head~tail对应一个非O实体
                        entity_set.add((head, tail))
                    head = tail = tail + 1
                    pass
                pre_suffix = suffix  # 移动处理下一个字

            else:  # 遇到了一句话的结尾
                if len(sentence.strip()):  # 句子不为空，处理并输出到文件
                    pass
                entity_set, sentence, pre_suffix, head, tail = self.init_sentence()


