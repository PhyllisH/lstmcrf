import tensorflow as tf

class JOINT_DENSE:
    def __init__(self, sen_len, num_hidden, ner_num, sa_num, vocab_size, embedding_size, embedding, learning_rate):
        self.input = tf.placeholder(tf.int32, [None, sen_len], name = 'input')
        self.ner = tf.placeholder(tf.int32, [None, sen_len], name = 'ner')
        self.sa = tf.placeholder(tf.int32, [None, sen_len], name = 'sa')
        self.dropout = tf.placeholder(tf.bool, name = 'drop')

        embeddings = tf.get_variable('embedding', shape=embedding.shape, initializer=tf.constant_initializer(embedding))

        wner = tf.get_variable(name='wner', shape=[num_hidden*2, ner_num], initializer=tf.truncated_normal_initializer(stddev=0.01), regularizer=tf.contrib.layers.l2_regularizer(0.001))
        bner = tf.Variable(tf.zeros([ner_num]), name='bner')

        wsa = tf.get_variable(name='wsa', shape=[num_hidden*2, sa_num], initializer=tf.truncated_normal_initializer(stddev=0.01), regularizer=tf.contrib.layers.l2_regularizer(0.001))
        bsa = tf.Variable(tf.zeros([sa_num]), name='bsa')

        vec = tf.nn.embedding_lookup(embeddings, self.input)
        self.sen_lens = self.length(self.input)
        sen_lens_ = tf.cast(self.sen_lens, tf.int64)
        vec = tf.cond(self.dropout, lambda:tf.nn.dropout(vec, 0.5), lambda:vec)

        self.trans_ner = tf.Variable(tf.zeros([ner_num, ner_num]), name='n')
        self.trans_sa = tf.Variable(tf.zeros([sa_num, sa_num]), name='s')
        forward, _ = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.LSTMCell(num_hidden),
            vec,
            dtype = tf.float32,
            sequence_length = self.sen_lens,
            scope='rnn_forward'
        )
        backward_, _ = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.LSTMCell(num_hidden),
            inputs = tf.reverse_sequence(vec, sen_lens_, seq_dim = 1),
            dtype = tf.float32,
            sequence_length = self.sen_lens,
            scope='rnn_backward'
        )
        backward = tf.reverse_sequence(backward_, sen_lens_, seq_dim = 1)
        output = tf.reshape(tf.concat(2, [forward, backward]), [-1, num_hidden*2])
        
        output_ner = tf.batch_matmul(output, wner) + bner
        self.unary_ner_score = tf.reshape(output_ner, [-1, sen_len, ner_num])
        lld_ner, self.trans_ner = tf.contrib.crf.crf_log_likelihood(self.unary_ner_score, self.ner, self.sen_lens, self.trans_ner)

        output_sa = tf.batch_matmul(output, wsa) + bsa
        self.unary_sa_score = tf.reshape(output_sa, [-1, sen_len, sa_num])
        lld_sa, self.trans_sa = tf.contrib.crf.crf_log_likelihood(self.unary_sa_score, self.sa, self.sen_lens, self.trans_sa)

        self.loss = tf.reduce_mean(-(lld_ner + lld_sa))

        self.trains = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices = 1)
        length = tf.cast(length, tf.int32)
        return length

