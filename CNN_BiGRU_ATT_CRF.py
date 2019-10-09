
import tensorflow as tf
from model import BiLSTM_CRF


class CNN_BiGRU_ATT_CRF(BiLSTM_CRF):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        super().__init__(args, embeddings, tag2label, vocab, paths, config)

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()  # 定义embedding层的变量（变量作用域"word")
        self.representation_layer_op()
        self.boundary_op(self.cnn_output)
        rnn_input = self.cnn_output
        if self.boundary:
            rnn_input = tf.concat([rnn_input, self.boundary_output], axis=-1) # batch, maxtime, hidden + 2*boundary_embedding
        self.biLSTM_layer_op(rnn_input)

        self.attention_layer_op()
        self.softmax_pred_op()
        self.logit_op(self.batch_att_res)
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        """  在基类的基础上补上自己的placeholder定义 """
        #self.window_size = tf.constant(value=5, dtype=tf.int32, name="window_size")
        super().add_placeholders()

    def lookup_layer_op(self):
        """ 定义embedding层的变量（变量作用域"word") """
        super().lookup_layer_op()

    def representation_layer_op(self):
        """ 使用cnn对输入的ebedding进行卷积，生成近似的词表示 """
        with tf.variable_scope("char_representation"):
            self.char_cnn_W_1 = tf.get_variable(name="cnn_W1",
                                           shape=[1, self.embedding_dim, self.hidden_dim],
                                           initializer=tf.contrib.layers.xavier_initializer(),
                                           dtype=tf.float32)
            self.char_cnn_W_3 = tf.get_variable(name="cnn_W3",
                                           shape=[3, self.embedding_dim, self.hidden_dim],
                                           initializer=tf.contrib.layers.xavier_initializer(),
                                           dtype=tf.float32)
            #self.char_cnn_W_5 = tf.get_variable(name="cnn_W5",
            #                               shape=[5, self.embedding_dim, self.hidden_dim],
             #                              initializer=tf.contrib.layers.xavier_initializer(),
             #                              dtype=tf.float32)
            #in: batch, max_seq, embedding_dim, 则kernel = [subseqlen, embedding_dim, output_channal]
            char_cnn_1 = tf.expand_dims(tf.tanh(tf.nn.conv1d(self.word_embeddings, self.char_cnn_W_1, stride=1, padding="SAME")), 1)
            char_cnn_3 = tf.expand_dims(tf.tanh(tf.nn.conv1d(self.word_embeddings, self.char_cnn_W_3, stride=1, padding="SAME")), 1)
            #char_cnn_5 = tf.expand_dims(tf.tanh(tf.nn.conv1d(self.word_embeddings, self.char_cnn_W_5, stride=1, padding="SAME")), 1)
            pooling_res = tf.reshape(
                                tf.nn.max_pool(tf.concat([char_cnn_1, char_cnn_3], 1),#, char_cnn_5], 1),
                                                ksize=[1, 2, 1, 1], strides=[1, 1, 1, 1], padding="VALID") # out-> batch, 1, max_seq, hidden_dim
                                , [self.var_batch_size, self.max_length, self.hidden_dim])
            self.cnn_output = tf.nn.dropout(pooling_res, self.dropout_pl)  # dropout后得到输出

    def biLSTM_layer_op(self, lstm_input):
        super().biLSTM_layer_op(lstm_input)

    def logit_op(self, logit_input):
        super().logit_op(logit_input)

    def window_attention_layer_op(self):
        with tf.variable_scope("cnn_att"):
            tf.Variable()
            self.cnn_W_q = tf.get_variable(name="W_q",
                                shape=[self.embedding_dim, self.hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            self.W_k = tf.get_variable(name="W_k",
                                  shape=[self.embedding_dim, self.hidden_dim],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype=tf.float32)
            self.V = tf.get_variable(name="V",
                                  shape=[self.hidden_dim, 1],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype=tf.float32)

            cond = lambda seq_len, sen_query, iter_idx, att_res: tf.cond(iter_idx + self.window_size//2 < seq_len, lambda: True, lambda: False)

            max_len = tf.shape(self.word_embeddings)[1]  # （batch, maxlen, embedding_dim)
            zero_padding = tf.zeros([1, self.window_size // 2, self.embedding_dim], dtype=tf.int32)
            att_repr = []
            for batch in range(self.batch_size):
                seq_len = self.sequence_lengths[batch]
                sen_query = self.word_embeddings[batch, : seq_len, :]  # 一句话的所有字
                iter_idx = tf.constant(0, dtype=tf.int32)
                att_res = zero_padding
                seq_len, sen_query, iter_idx, att_res = tf.while_loop(cond, self.local_attention_loop_body, [seq_len, sen_query, iter_idx, att_res])
                att_repr.append(self.compute_attention_weight(query, query, seq_len, max_len))
            self.att_repr = tf.stack(att_repr)
            #self.att_repr = self.compute_batch_attention_weight(query=self.rnn_output, max_len=max_len, batch_size=batch_size)
            print(self.att_repr.shape)

    def local_attention_loop_body(self, seq_len, sen_query, iter_idx, att_res):
        left = tf.cond(iter_idx < self.window_size // 2, lambda: 0, lambda: iter_idx - self.window_size // 2)
        right = tf.cond(iter_idx + self.window_size // 2 < seq_len, lambda: iter_idx + self.window_size // 2, lambda: seq_len)
        query = sen_query[iter_idx]
        key = sen_query[left: right, :]
        # 前面/后面不足，补0,  确保构造出( window_size, embedding_dim)
        key = tf.cond(iter_idx < self.window_size // 2, lambda: tf.concat([tf.zeros([self.window_size // 2 - iter_idx, self.embedding_dim], dtype=tf.int32), query], 0),
                        lambda: key)

        key = tf.cond(iter_idx + self.window_size // 2 < seq_len, lambda: key,
                        lambda: tf.concat([query, tf.zeros([self.window_size // 2 + iter_idx - seq_len, self.embedding_dim], dtype=tf.int32)], 0),)

        iter_idx += 1
        return seq_len, sen_query, iter_idx, att_res

    def compute_local_attention(self, query, key):
        linear = tf.matmul(query, self.W_q) + tf.matmul(key, self.W_k)  # (window_size , h_dim)
        score = tf.matmul(tf.nn.tanh(linear), self.V)  # (window_size, 1) 即query和每个key的score值
        score = tf.reshape(score, [self.window_size])
        alpha = tf.nn.softmax(score)  # 计算att权重
        att_repre = alpha * key  # att = alpha*xj
        return att_repre

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

            batch_cnt = tf.constant(0, dtype=tf.int32)
            batch_att_res = tf.zeros([1, self.max_length, 2 * self.hidden_dim])  # 先初始化一个0行，之后再去掉
            loop_con = lambda cnt, _: tf.cond(cnt < self.var_batch_size, lambda: True, lambda: False)
            for batch in range(self.batch_size):
                batch_cnt, batch_att_res = tf.while_loop(loop_con, self.compute_attention_weight, [batch_cnt, batch_att_res], shape_invariants=[batch_cnt.shape, tf.TensorShape([None, None, 2*self.hidden_dim])])
            self.batch_att_res = tf.nn.dropout(batch_att_res[1:],self.dropout_pl)

    def compute_attention_weight(self, batch, loop_res):
        seq_len = self.sequence_lengths[batch]
        query = self.rnn_output[batch, : seq_len, :]
        linear = tf.expand_dims(tf.matmul(query, self.W_q), 1) + tf.expand_dims(tf.matmul(query, self.W_k), 0)  #(seqlen, seqlen , 2*h_dim)
        #print("linear's shape is {}，Q_exp's shape is {}, K_exp's shape is{}, seq_len is{}. max_len is {}".format(linear.shape, Q_exp.shape,
        #                                                                            K_exp.shape, seq_len, max_len))
        score = tf.sigmoid(linear)   # element-wise, (seqlen, seqlen , 2*h_dim)
        alpha_ij = tf.multiply(score, tf.expand_dims(query, 1))  # (seqlen, seqlen , 2*h_dim)
        alpha = tf.divide(tf.reshape(tf.reduce_sum(alpha_ij, 1), [seq_len, 2 * self.hidden_dim]), tf.to_float(seq_len))  # (seqlen, 1 , 2*h_dim) -> (seqlen, 2*hid_dim)

        padding = tf.zeros([self.max_length - seq_len, 2 * self.hidden_dim], dtype=tf.float32)
        att_repre = tf.expand_dims(tf.concat([tf.tanh(alpha), padding], axis=0), 0)  # 1, maxlen, hdim

        loop_res = tf.concat([loop_res, att_repre], 0)  # 这句话的att结果拼上去
        batch += 1
        return batch, loop_res
