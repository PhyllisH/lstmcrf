import tensorflow as tf

class COMBINE():
    def __init__(self, sen_len, num_hidden, label_num, vocab_size, embedding_size, embedding, learning_rate, sparse_num, crf_num):
        self.input = tf.placeholder(tf.int32, [None, sen_len], name='input')
        self.sparse = tf.placeholder(tf.int32, [None, sen_len, sparse_num], name='sparse')
        self.label = tf.placeholder(tf.int32, [None, sen_len], name='label')
        self.dropout = tf.placeholder(tf.bool, name = 'drop')

        embeddings = tf.get_variable('embedding', shape=embedding.shape, initializer=tf.constant_initializer(embedding))
        who = tf.get_variable(name='who', shape=[num_hidden*2, label_num], initializer=tf.truncated_normal_initializer(stddev=0.01), regularizer=tf.contrib.layers.l2_regularizer(0.001))
        bho = tf.Variable(tf.zeros([label_num]), name='bias')
        pa = tf.get_variable('pa', shape = [crf_num, num_hidden], initializer = tf.truncated_normal_initializer(stddev=0.01))
        wfo = tf.get_variable(name='wfo', shape=[num_hidden*2, label_num], initializer=tf.truncated_normal_initializer(stddev=0.01), regularizer=tf.contrib.layers.l2_regularizer(0.001))
        bfo = tf.Variable(tf.zeros([label_num]), name='biasf')

        vec_word = tf.nn.embedding_lookup(embeddings, self.input)
        self.sen_lens = self.length(self.input)
        sen_lens_ = tf.cast(self.sen_lens, tf.int64)
        vec_word = tf.cond(self.dropout, lambda: tf.nn.dropout(vec_word, 0.5), lambda:vec_word)

        vec_sparse = tf.nn.embedding_lookup(pa, self.sparse)
        mask = tf.reshape(tf.sign(self.sparse), [-1])
        vec_sparse = tf.reshape(tf.transpose(tf.transpose(tf.reshape(vec_sparse, [-1, num_hidden])) * tf.cast(mask, tf.float32)), [-1, sen_len, sparse_num, num_hidden])
        vec_sparse_ = tf.reduce_mean(vec_sparse, 2)
        #vec_sparse_ = tf.cond(self.dropout, lambda: tf.nn.dropout(vec_sparse_, 0.5), lambda: vec_sparse_)

        forward_dense, _ = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.LSTMCell(num_hidden),
            vec_word,
            dtype = tf.float32,
            sequence_length = self.sen_lens,
            scope = 'rnn_forward_dense'
        )
        backward_dense_, _ = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.LSTMCell(num_hidden),
            inputs = tf.reverse_sequence(vec_word, sen_lens_, seq_dim = 1),
            dtype = tf.float32,
            sequence_length = self.sen_lens,
            scope = 'rnn_backward_dense'
        )
        backward_dense = tf.reverse_sequence(backward_dense_, sen_lens_, seq_dim = 1)
        output_dense = tf.reshape(tf.concat(2, [forward_dense, backward_dense]), [-1, num_hidden * 2])
        output_dense_ = tf.batch_matmul(output_dense, who) + bho
        unary_dense_score = tf.reshape(output_dense_, [-1, sen_len, label_num])
        forward_sparse, _ = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.LSTMCell(num_hidden),
            vec_sparse_,
            dtype = tf.float32,
            sequence_length = self.sen_lens,
            scope = 'rnn_forward_sparse'
        )
        backward_sparse_, _ = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.LSTMCell(num_hidden),
            inputs = tf.reverse_sequence(vec_sparse_, sen_lens_, seq_dim = 1),
            dtype = tf.float32,
            sequence_length = self.sen_lens,
            scope = 'rnn_backward_sparse'
        )
        backward_sparse = tf.reverse_sequence(backward_sparse_, sen_lens_, seq_dim = 1)
        output_sparse = tf.reshape(tf.concat(2, [forward_sparse, backward_sparse]), [-1, num_hidden * 2])
        output_sparse_ = tf.batch_matmul(output_sparse, wfo) + bfo
        unary_sparse_score = tf.reshape(output_sparse_, [-1, sen_len, label_num])
        self.unary_score = tf.add(unary_dense_score, unary_sparse_score)
        lld, self.trans = tf.contrib.crf.crf_log_likelihood(self.unary_score, self.label, self.sen_lens)
        self.loss = tf.reduce_mean(-lld)
        self.trains = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices = 1)
        length = tf.cast(length, tf.int32)
        return length

