#-*-encoding:utf8-*-#
import sys, pickle, os, random
import numpy as np

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data -- 句子数个二元组，每个二元组( [字list]， [tag list] )
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    begin_, end_ = [], []
    for line in lines:
        if line != '\n':
            line = line.strip()
            try:
                [char, label] = line.split(" ")
            except:
                if line[0:2] == "  ":
                    char, label = " ", line[2:]
            sent_.append(char.lower())
            tag_.append(label)
            begin_.append(1) if label[0] == "B" else begin_.append(0)
        else:
            for tdx in range(len(tag_)):
                tag = tag_[tdx]
                end_.append(1) if tag != "O" and (tdx + 1 == len(tag_) or tag_[tdx + 1] == "O") else end_.append(0)

            data.append((sent_, tag_, begin_, end_))
            sent_, tag_, begin_, end_ = [], [], [], []

    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """
    扫描输入文件，统计字频，构造word2id字典，对小于min_count的字按 unknown处理
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def count_oov(word2id, data, log_path, type="train"): ## 统计数据中的oov数量
    oov_set = set()
    char_set = set()
    for sen_, _, _, _ in data:
        for word in sen_:
            char_set.add(word)
            if word not in word2id:
                oov_set.add(word)
    from utils import get_logger
    get_logger(log_path).debug("%s数据中总共有%d个不同的字符，其中oov字数共有%d个，占比%.4f" %
                 (type, len(char_set), len(oov_set), len(oov_set) * 100.0/len(char_set)))
    oov_str = ""
    for word in oov_set:
        oov_str += "," + word
    print(oov_str)


"""  2019-5-22:为了能够和带有单个数字/字母的pretrain char embedding兼容，调整代码结构，在没有预训练词向量的时候保留原功能"""
def sentence2id(sent, word2id, unk='<UNK>'):
    """
    字转id，其中数字一律以数字标签处理，英文一律以英文标签处理，
    :param sent:
    :param word2id:
    :return: 字的id list
    """
    sentence_id = []
    for word in sent:
        if unk == '<UNK>':
            if word not in word2id:
                if word.isdigit():
                    word = '<NUM>'
                elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                    word = '<ENG>'
                else:
                    word = '<UNK>'
        else:
            if word not in word2id:
                word = unk
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path, unk_mark):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    has_unk = True
    if unk_mark not in word2id:
        word2id[unk_mark] = len(word2id)
        has_unk = False
    return word2id, has_unk


def random_embedding(vocab, embedding_dim):
    u"""
    随机初始化一个均匀分布[-0.25, 0.25]，shape为(字典长*embedding_dim)
    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """
    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences)) # 根据当前batch的字id列表的长度，计算当前batch的最大句子长度
    #max_len = min(max_len, 300)
    seq_list, seq_len_list = [], []
    ## 对输入字id的list做zero padding，但是保留每个句子的实际长度 （只是为了创建一个batch的张量不失败？）
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list, max_len


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False, unk='<UNK>'):
    """
    处理训练数据为batch数据，包括：句子顺序打乱、标签映射到id，字映射到id
    数字与英文均分别处理为统一标识符
    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels, begins, ends = [], [], [], []
    for (sent_, tag_, begin_, end_) in data:
        sent_ = sentence2id(sent_, vocab, unk) # 句子里的每个字的id构成的list
        label_ = [tag2label[tag] for tag in tag_] # 句子里每个字的tag的id构成的list

        if len(seqs) == batch_size:
            yield seqs, labels, begins, ends # 积累的数据达到batch_size，返回当前积累的数据，并清空当前batch，下次继续
            seqs, labels, begins, ends = [], [], [], []

        seqs.append(sent_)
        labels.append(label_)
        begins.append(begin_)
        ends.append(end_)

    if len(seqs) != 0:
        #seqs.extend([[]] * (batch_size-len(seqs)))
        #labels.extend([[]] * (batch_size - len(labels)))
        yield seqs, labels, begins, ends

