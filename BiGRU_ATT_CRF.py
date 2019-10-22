import tensorflow as tf
from model import BiLSTM_CRF

"""
这个最近没用，有些地方跟基类可能对不上了，之后修吧=、=
"""
class BiGRU_ATT_CRF(BiLSTM_CRF):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        super().__init__(args, embeddings, tag2label, vocab, paths, config)

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op() # 定义embedding层的变量（变量作用域"word")
        self.biLSTM_layer_op(self.word_embeddings)
        self.attention_layer_op()
        self.softmax_pred_op()
        rnn_att_out = tf.concat([self.rnn_output, self.batch_att_res], axis=-1)  # 记得跳过第一段
        rnn_att_out = tf.nn.dropout(rnn_att_out, self.dropout_pl)  # BiLSTM的结果过个dropout层
        self.logit_op(rnn_att_out)
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        super().add_placeholders()

    def logit_op(self, logit_input):
        super().logit_op(logit_input)

    def attention_layer_op(self):
        with tf.variable_scope("att"):
            self.W_q = tf.get_variable(name="W_q",
                                shape=[2 * self.hidden_dim, 2 * self.hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            self.W_k = tf.get_variable(name="W_k",
                                  shape=[2 * self.hidden_dim, 2 * self.hidden_dim],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype=tf.float32)
            """self.V = tf.get_variable(name="V",
                      shape=[2 * self.hidden_dim, 1],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      dtype=tf.float32)"""

            batch_cnt = tf.constant(0, dtype=tf.int32)
            batch_att_res = tf.zeros([1, self.max_length, 2 * self.hidden_dim])  # 先初始化一个0行，之后再去掉
            loop_con = lambda cnt, _: tf.cond(cnt < self.var_batch_size, lambda: True, lambda: False)
            for batch in range(self.batch_size):
                batch_cnt, batch_att_res = tf.while_loop(loop_con, self.compute_attention_weight, [batch_cnt, batch_att_res], shape_invariants=[batch_cnt.shape, tf.TensorShape([None, None, 2*self.hidden_dim])])
                #att_repr.append(self.compute_attention_weight(query, query, seq_len, max_len))
            #self.att_repr = tf.stack(att_repr)
            #self.att_repr = self.compute_batch_attention_weight(query=self.rnn_output, max_len=max_len, batch_size=batch_size)
            #print(self.att_repr.shape)
            self.batch_att_res = batch_att_res[1:]

    def compute_batch_attention_weight(self, query, max_len, batch_size):
        query = tf.reshape(query, [batch_size * max_len, 2 * self.hidden_dim])  # (batch*maxlen, 2*hidden)
        Q_exp = tf.reshape(tf.matmul(query, self.W_q), [batch_size, max_len, 2 * self.hidden_dim])  # (b, l , 2*hidden)
        K_exp = tf.reshape(tf.matmul(query, self.W_k), [batch_size, max_len, 2 * self.hidden_dim])  # (b, l , 2*hidden)
        linear = tf.reshape(tf.expand_dims(Q_exp, 2) + tf.expand_dims(K_exp, 1), [-1, 2 * self.hidden_dim])  #(bacth_size*max_len*max_len , 2*h_dim)
        print("linear's shape is {}，Q_exp's shape is {}, K_exp's shape is{}, max_len is {}".format(linear.shape, Q_exp.shape,
                                                                                    K_exp.shape, max_len))
        score = tf.matmul(tf.nn.tanh(linear), self.V)  # (bacth_size*max_len*max_len, 1) 即query和每个key的score值
        alpha = tf.nn.softmax(tf.reshape(score, [batch_size, max_len, max_len]), axis=1)  # 计算att权重, (bacth_size, max_len, max_len)
        print("linear's shape is {}，score's shape is {}, alpha's shape is{}".format(linear.shape, score.shape, alpha.shape))
        #print("alpha = {}, sum={}".format(alpha, tf.reduce_sum(alpha)))
        query = tf.reshape(query, [batch_size, max_len, 2 * self.hidden_dim])  # (batch*maxlen, 2*hidden)
        att_repre = tf.matmul(alpha, query)  # (batch, maxlen,maxlen)*(batch, maxlen,hdim) = batchm maxlen, hdim

        return att_repre

    def compute_attention_weight(self, batch, loop_res):
        seq_len = self.sequence_lengths[batch]
        query = self.rnn_output[batch, : seq_len, :]
        linear = tf.expand_dims(tf.matmul(query, self.W_q), 1) + tf.expand_dims(tf.matmul(query, self.W_k), 0)  #(seqlen, seqlen , 2*h_dim)
        #print("linear's shape is {}，Q_exp's shape is {}, K_exp's shape is{}, seq_len is{}. max_len is {}".format(linear.shape, Q_exp.shape,
        #                                                                            K_exp.shape, seq_len, max_len))
        score = tf.sigmoid(linear)   # element-wise, (seqlen, seqlen , 2*h_dim)
        alpha_ij = tf.multiply(score, tf.expand_dims(query, 1))  # (seqlen, seqlen , 2*h_dim)
        alpha = tf.divide(tf.squeeze(tf.reduce_sum(alpha_ij, 1)), tf.to_float(seq_len))  # (seqlen, 1 , 2*h_dim) -> (seqlen, 2*hid_dim)

        padding = tf.zeros([self.max_length - seq_len, 2 * self.hidden_dim], dtype=tf.float32)
        att_repre = tf.expand_dims(tf.concat([tf.tanh(alpha), padding], axis=0), 0)  # 1, maxlen, hdim

        loop_res = tf.concat([loop_res, att_repre], 0)  # 这句话的att结果拼上去
        batch += 1
        return batch, loop_res

    def compute_attention_weight_2(self, batch, loop_res):
        seq_len = self.sequence_lengths[batch]
        query = self.rnn_output[batch, : seq_len, :]
        linear = tf.matmul(query, self.W_q)  # (l, 2*h) (2*h, 2*h)  => l, 2*h
        linear = tf.matmul(linear, tf.transpose(query)) # => l * l

        alpha = tf.nn.softmax(linear, axis=1)  # 计算att权重, (seqlen, seqlen)
        #print("linear's shape is {}，score's shape is {}, alpha's shape is{}".format(linear.shape, score.shape, alpha.shape))
        #print("alpha = {}, sum={}".format(alpha, tf.reduce_sum(alpha)))
        att_repre = tf.matmul(alpha, query)  # (seqlen,seqlen)*(seqlen,hdim) = seqlen*hdim

        padding = tf.zeros([self.max_length - seq_len, 2 * self.hidden_dim], dtype=tf.float32)
        att_repre = tf.expand_dims(tf.concat([att_repre, padding], axis=0), 0) # 1, maxlen, hdim
        batch += 1
        loop_res = tf.concat([loop_res, att_repre], 0) # 这句话的att结果拼上去
        return batch, loop_res

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