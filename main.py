#-*-encoding:utf8-*-#
import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
from BiGRU_ATT_CRF import BiGRU_ATT_CRF
from CNN_BiGRU_ATT_CRF import CNN_BiGRU_ATT_CRF
from utils import str2bool, get_logger, get_entity, get_multiple_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding, count_oov


## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # 按需分配GPU
config.gpu_options.per_process_gpu_memory_fraction = 1.0  # 分配固定大小最多占显存的0.2 need ~700MB GPU memory


## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
parser.add_argument('--unk', type=str, default='<UNK>', help='the tag for unknown word when a word is missing in the word2id')
parser.add_argument('--word2id', type=str, default='word2id.pkl', help='word2id file name(same dir with the train_data)')
parser.add_argument('--restore', type=str2bool, default=False, help='use exisiting checkpoint.')
parser.add_argument('--rho', type=float, default=0.02, help='learning rate decrease speed by each epoch')
args = parser.parse_args()


## get char embeddings
u""" 读取预处理的word2id文件（实际上是每个字分配一个id) """
word2id = read_dictionary(os.path.join('.', args.train_data, args.word2id))

u""" 随机初始化或者加载预训练的字符embedding """
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = os.path.join('.', args.train_data, args.pretrain_embedding)
    embeddings = np.array(np.load(embedding_path), dtype='float32')
    print(embeddings.shape)
    args.embedding_dim = embeddings.shape[1]
    #args.hidden_dim = args.embedding_dim  # 修正hidden_state的长度


## read corpus and get training data
if args.mode != 'demo':
    train_path = os.path.join('.', args.train_data, 'train_data')
    test_path = os.path.join('.', args.test_data, 'test_data')
    """ 取出训练数据、测试数据，格式： 句子数个二元组，每个二元组( [字list]， [tag list] ) """
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path); test_size = len(test_data)

## paths setting
""" 处理对模型结果等文件的保存名字及路径, 以及logger的保存位置 """
paths = {}
timestamp = 'test'#time.strftime("%Y%m%d%H%M", time.localtime()) if args.mode == 'train' else args.demo_model
args.demo_model = timestamp
output_path = os.path.join('.', args.train_data+"_save", timestamp)  # 模型保存的目录 + 时间戳作为模型的名字
if not os.path.exists(output_path): os.makedirs(output_path)  # 目录不存在则创建对应目录

summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)

model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix

result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)

log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))

##当存在label更新时，更新label
label_path = os.path.join('.', args.train_data, "labels_data")
if os.path.exists(label_path):
    import json
    with open(label_path, "r") as label_file:
        tag2label = json.load(label_file)
        get_logger(log_path).info("检测到label文件更新，更新后的label数量为：%d个" % (len(tag2label)))


## training model
if args.mode == 'train':
    count_oov(word2id, train_data, log_path, type="train_data")  # 统计输出oov
    count_oov(word2id, test_data, log_path, type="test_data")

    model = CNN_BiGRU_ATT_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    ## hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    ## train model on the whole training data
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    #model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiGRU_ATT_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while(1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                #PER, LOC, ORG = get_entity(tag, demo_sent)
                #print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))
                entity_dict = get_multiple_entity(tag, demo_sent)
                res_str = ""
                for entity in entity_dict:
                    res_str += '{}: {}\n'.format(entity, entity_dict[entity])
                print(res_str)
