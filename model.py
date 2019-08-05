import numpy as np
import os, time, sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield
from utils import get_logger
from eval import conlleval


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.CRF = args.CRF  # CRF使能
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab  # 字典：字->id
        self.shuffle = args.shuffle
        self.unk = args.unk
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op() # 定义embedding层的变量（变量作用域"word")
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        """ 为所有的输入变量设置placeholder """
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")  # 推测形状：batch*sentlen
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")  # 推测形状：batch*sentlen
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")  # 推测形状：batch
        """ 两个超参数 """
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr_pl")

    def lookup_layer_op(self):
        """ 定义embedding层的变量（变量作用域"word") """
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,  # 输入的loc
                                                     name="word_embeddings")
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout_pl)  # 套个dropout层后才是最后的embedding层

    def biLSTM_layer_op(self):
        """ 定义BiLSTM层的变量(变量作用域"bi-lstm") """
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim) # 前向LSTM单元 cell_fw
            cell_bw = LSTMCell(self.hidden_dim) # 后向LSTM单元 cell_bw
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            """
            	·输出为(output_fw, output_bw),  (output_fw_state, output_bw_state) 
            	即一个包含前向cell输出tensor和后向cell输出tensor组成的二元组。
            	在time_major=False的情形下，每个tensor的shape为（batch* max_time* depth）。depth是词向量的长度
				·output_fw_state, output_bw_state 则分别是两个方向最后的隐藏状态组成的二元组，类型为LSTMStateTuple(c, h)
                当句子长度<max_time时，fw_state中的h是output_fw张量中不全为零的最后一行（正向，反向的时候是第一行）
            """
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1) # 把正反向的输出在depth维度(hidden_state)上接起来concat
            output = tf.nn.dropout(output, self.dropout_pl) # BiLSTM的结果过个dropout层

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            """ BiLSTM的输出后接一层全连接，由[-1, 2*hidden_dim]->[-1, 7（num_tags）] """
            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags]) # 形状还原回 batch, maxtime, num_tags

    def loss_op(self):
        """CRF层+loss计算"""
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits, # 全连接的结果( batch, maxtime, num_tags)
                                                                   tag_indices=self.labels, # labels (batch, seqlen,tag_indice)
                                                                   sequence_lengths=self.sequence_lengths) # seqlen (batch, 1)
            print(self.transition_params.name)
            """对数似然估计值。由于似然函数本身是概率值，取值0~1，故对数化后就变成负无穷~0，但依然保持单调性。 """
            self.loss = -tf.reduce_mean(log_likelihood)  # 估计值是负数的。故累加到loss就是用减法
            # reduce_mean本身可以指定保留的维度。不指定的情况下计算所有维度的均值，得到一个标量
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss) # 参数可视化：显示这个 标量信息 loss

    def softmax_pred_op(self):
        """Softmax层（对比试验，当不使用CRF时才用）"""
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)  # 将损失传入优化器
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        """

        :param train:
        :param dev:
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)
            fb1 = -1.0
            for epoch in range(self.epoch_num):
                evaluate_dict, step_num = self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)
                # 每个epoch结束后，保存下当前最好的模型（按fb1取值是否更高）
                if evaluate_dict["FB1"] > fb1:
                    self.logger.info("FB1值取得新的最优值%.2f，保存模型"%evaluate_dict["FB1"])
                    saver.save(sess, self.model_path, global_step=epoch)
                    fb1 = evaluate_dict["FB1"]

    def test(self, test):
        saver = tf.train.import_meta_graph(self.model_path + ".meta")
        #saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)

    def demo_one(self, sess, sent):
        """
        直接预测，并对预测出来的序列进行转码
        :param sess:
        :param sent: 
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False, unk=self.unk):
            label_list_, _ = self.predict_one_batch(sess, seqs, demo=True)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        """

        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle, unk=self.unk)
        step_num = 1
        for step, (seqs, labels) in enumerate(batches):
            #sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob) # 为预定义的每个placeholder绑定数据，将当前batch的输入数据组织成list
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict) # session执行op运算，并获得对应op的返回值，其中loss和summary结果要显式使用
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)



        self.logger.info('===========validation / train===========')
        label_list_train, seq_len_list_train = self.dev_one_epoch(sess, train)
        self.evaluate(label_list_train, seq_len_list_train, train, epoch)

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        evaluate_dict = self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)
        return evaluate_dict, step_num

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """
        为预定义的每个placeholder绑定数据，将当前batch的输入数据组织成list
        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
        # 对输入字id的list做zero padding到此batch最大句子长度，但是保留每个句子的实际长度

        feed_dict = {"word_ids:0": word_ids,
                     "sequence_lengths:0": seq_len_list}
        if labels is not None: # dev / eval时可能为空
            labels_, _ = pad_sequences(labels, pad_mark=0) # 同上， 对label id 的list作 padding
            feed_dict["labels:0"] = labels_
        if lr is not None:
            feed_dict["lr_pl:0"] = lr
        if dropout is not None:
            feed_dict["dropout:0"] = dropout

        return feed_dict, seq_len_list # 在dev/eval时候要使用句子长度，参与计算CRF

    def dev_one_epoch(self, sess, dev):
        """
        将验证集用于目前的模型，计算得到对应的标注序列及每个句子的实际长度（用来算CRF）
        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False, unk=self.unk):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs, demo=False):
        """
        通过session调用网络计算到FC层为止，并取出模型目前的CRF转移矩阵
         使用转移矩阵对发射概率矩阵（全连接的结果）进行解码，返回评分最高的序列和序列的评分
        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:  # 使用CRF层的情况下，如果直接调用train_op，则CRF层的转移矩阵也会被参与计算，所以只能调用到全连接为止
            """ 通过session调用网络计算到FC层为止，并取出模型目前的CRF转移矩阵 """
            logits, transition_params = sess.run([tf.get_default_graph().get_tensor_by_name("proj/Reshape_1:0"),
                                                  tf.get_default_graph().get_tensor_by_name("transitions:0")],
                                                 feed_dict=feed_dict)
            if demo:
                print("发射矩阵如下", logits, logits.shape)
                print("转移概率矩阵如下", transition_params, transition_params.shape)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params) # 使用转移矩阵对发射概率矩阵（全连接的结果）进行解码，返回评分最高的序列
                if demo:
                    print("最优序列：", viterbi_seq, "得分为%.4f"%_)
                    self.print_viterbi_score(viterbi_seq, logit, transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def print_viterbi_score(self, viterbi_seq, logit, transition_params):
        j = 0
        sum = 0.
        sum_seq = "0"
        for i in range(len(viterbi_seq)):
            now = viterbi_seq[i]
            send_score = logit[j][now]
            sum_seq += "+ %s " % str(send_score)
            if i + 1 < len(viterbi_seq):
                next = viterbi_seq[i + 1]
                trans_score = transition_params[now][next]
                sum_seq += "+ %s " % str(trans_score)
                sum += trans_score
            sum += send_score
            j += 1
        sum_seq += "=%s" % str(sum)
        print ("score 计算式为：%s"%sum_seq)

    def cnt_all_score(self, depth, now, logit, transition_params):

        send_score = logit[depth][now]
        if depth + 1 < len(logit):
            for next in range(len(transition_params)):
                trans_score = transition_params[now][next]  # 从now状态转移到next状态


    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """
        检验预测序列结果和实际序列是否一致（生成结果的时候已经使用过了句子长度，已经没用了）
        对每个验证数据，生成[原始数据，原始标签，预测标签]三元组
        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label # {0:0, 1:"B-PER" ...}

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data): # 对每个验证数据，生成[原始数据，原始标签，预测标签]三元组
            tag_ = [label2tag[label__] for label__ in label_] # 将label转换回tag（O以外）
            sent_res = []
            if  len(label_) != len(sent): # 异常，特别输出长度对不上的情况
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)

        metrics = conlleval(model_predict, label_path, metric_path) # 调用脚本计算评估指标，第0行是token的情况，第1行是总体entity评估，余下行是逐个entity评估
        for _ in metrics:
            self.logger.info(_)
        return self.metric2dict(metrics[1])

    def metric2dict(self, evaluate):
        """
        对输入的形如“accuracy:  71.46%; precision:   0.00%; recall:   0.00%; FB1:   0.00”的字符串进行解析，生成dict类型对象，供使用
        :param metric_str:
        :return:
        """
        evaluate = evaluate.replace(" ", "").split(";")
        assert(len(evaluate) == 4)
        evaluate_dict = {}
        for indice in evaluate:
            [indice_key, indice_value] = indice.split(":")
            indice_value = float(indice_value.replace("%", "")) # 可能有百分号导致不能转换
            evaluate_dict[indice_key] = indice_value
        return evaluate_dict


