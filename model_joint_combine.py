import tensorflow as tf

class JOINT_COMBINE:
    def __init__(self, batch_size, sen_len, num_hidden, ner_num, sa_num, vocab_size, embedding_size, embedding, learning_rate, sparse_num, crf_num):
        self.input = tf.placeholder(tf.int32, [None, sen_len], name = 'input')
        self.sparse = tf.placeholder(tf.int32, [None, sen_len, sparse_num], name = 'sparse')
        self.ner = tf.placeholder(tf.int32, [None, sen_len], name = 'ner')
        self.sa = tf.placeholder(tf.int32, [None, sen_len], name = 'sa')
        self.dropout = tf.placeholder(tf.bool, name = 'drop')

        embeddings = tf.get_variable('embedding', shape = embedding.shape, initializer = tf.constant_initializer(embedding))
        pa = tf.get_variable('pa', shape = [crf_num, num_hidden], initializer = tf.truncated_normal_initializer(stddev=0.01), regularizer = tf.contrib.layers.l2_regularizer(0.001))
        
        wner_w = tf.get_variable(name = 'wner_w', shape = [num_hidden * 2, ner_num], initializer = tf.truncated_normal_initializer(stddev = 0.01), regularizer = tf.contrib.layers.l2_regularizer(0.001))
        bner_w = tf.Variable(tf.zeros([ner_num]), name = 'bner_w')
        wner_s = tf.get_variable(name = 'wner_s', shape = [num_hidden * 2, ner_num], initializer = tf.truncated_normal_initializer(stddev = 0.01), regularizer = tf.contrib.layers.l2_regularizer(0.001))
        bner_s = tf.Variable(tf.zeros([ner_num]), name = 'bner_s')

        wsa_w = tf.get_variable(name = 'wsa_w', shape = [num_hidden * 2, sa_num], initializer = tf.truncated_normal_initializer(stddev = 0.01), regularizer = tf.contrib.layers.l2_regularizer(0.001))
        bsa_w = tf.Variable(tf.zeros([sa_num]), name = 'bsa_w')
        wsa_s = tf.get_variable(name = 'wsa_s', shape = [num_hidden * 2, sa_num], initializer = tf.truncated_normal_initializer(stddev = 0.01), regularizer = tf.contrib.layers.l2_regularizer(0.001))
        bsa_s = tf.Variable(tf.zeros([sa_num]), name = 'bsa_s')

        self.trans_ner = tf.Variable(tf.zeros([ner_num, ner_num]), name='n')
        self.trans_sa = tf.Variable(tf.zeros([sa_num, sa_num]), name='s')

        vec_word = tf.nn.embedding_lookup(embeddings, self.input)
        self.sen_lens = self.length(self.input)
        vec_word = tf.cond(self.dropout, lambda:tf.nn.dropout(vec_word, 0.5), lambda:vec_word)

        vec_sparse = tf.nn.embedding_lookup(pa, self.sparse)
        mask = tf.reshape(tf.sign(self.sparse), [-1])
        vec_sparse = tf.reshape(tf.transpose(tf.transpose(tf.reshape(vec_sparse, [-1, num_hidden])) * tf.cast(mask, tf.float32)), [-1, sen_len, sparse_num, num_hidden])
        vec_sparse = tf.reduce_mean(vec_sparse, 2)

        output_w, _ = tf.nn.bidirectional_dynamic_rnn(
            tf.nn.rnn_cell.LSTMCell(num_hidden), 
            tf.nn.rnn_cell.LSTMCell(num_hidden), 
            vec_word, 
            dtype = tf.float32,
            sequence_length = self.sen_lens,
            scope = 'brnn_word'
        )
        output_w = tf.reshape(tf.concat(2, output_w), [-1, num_hidden*2])
        output_w_ner = tf.batch_matmul(output_w, wner_w) + bner_w
        unary_w_ner_score = tf.reshape(output_w_ner, [-1, sen_len, ner_num])
        output_w_sa = tf.batch_matmul(output_w, wsa_w) + bsa_w
        unary_w_sa_score = tf.reshape(output_w_sa, [-1, sen_len, sa_num])

        output_s, _ = tf.nn.bidirectional_dynamic_rnn(
            tf.nn.rnn_cell.LSTMCell(num_hidden), 
            tf.nn.rnn_cell.LSTMCell(num_hidden), 
            vec_sparse, 
            dtype = tf.float32,
            sequence_length = self.sen_lens,
            scope = 'brnn_sparse'
        )
        output_s = tf.reshape(tf.concat(2, output_s), [-1, num_hidden*2])
        output_s_ner = tf.batch_matmul(output_s, wner_s) + bner_s
        unary_s_ner_score = tf.reshape(output_s_ner, [-1, sen_len, ner_num])
        output_s_sa = tf.batch_matmul(output_s, wsa_s) + bsa_s
        unary_s_sa_score = tf.reshape(output_s_sa, [-1, sen_len, sa_num])
        self.unary_ner_score = tf.add(unary_w_ner_score, unary_s_ner_score)
        self.unary_sa_score = tf.add(unary_w_sa_score, unary_s_sa_score)
        lld_ner, self.trans_ner = tf.contrib.crf.crf_log_likelihood(self.unary_ner_score, self.ner, self.sen_lens, self.trans_ner)
        lld_sa, self.trans_sa = tf.contrib.crf.crf_log_likelihood(self.unary_sa_score, self.sa, self.sen_lens, self.trans_sa)
        self.loss = tf.reduce_mean(-(lld_ner + lld_sa))
        self.trains = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices = 1)
        length = tf.cast(length, tf.int32)
        return length

