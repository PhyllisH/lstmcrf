import tensorflow as tf
from tensorflow.contrib import learn

class LSTM():
    def __init__(self, sen_len, num_hidden, label_num, vocab_size, embedding_size, embedding, learning_rate, label_m):
        self.input = tf.placeholder(tf.int32, [None, sen_len], name='input')
        self.label = tf.placeholder(tf.int32, [None, sen_len], name='label')
	self.dropout = tf.placeholder(tf.bool, name = 'drop')
        who = tf.get_variable(name='who', shape=[num_hidden*2, label_num], initializer=tf.truncated_normal_initializer(stddev=0.01), regularizer=tf.contrib.layers.l2_regularizer(0.001))
        embeddings = tf.get_variable('embedding', shape=embedding.shape, initializer=tf.constant_initializer(embedding))
        bho = tf.Variable(tf.zeros([label_num]), name='bias')
        #who = tf.Variable(tf.random_normal([num_hidden*2, label_num], stddev = 0.05),name="who")
	param = tf.get_variable('param', shape = label_m.shape, initializer=tf.constant_initializer(label_m))
	plabel = tf.nn.embedding_lookup(param, self.label)
        vec = tf.nn.embedding_lookup(embeddings, self.input)
        self.sen_lens = self.length(self.input)
        sen_lens_ = tf.cast(self.sen_lens, tf.int64)
        #if self.dropout == 1:
	vec = tf.cond(self.dropout, lambda:tf.nn.dropout(vec, 0.5), lambda:vec)
	#vec = tf.nn.dropout(vec, 0.5)
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
        output_ = tf.batch_matmul(output, who) + bho
        self.unary_score = tf.reshape(output_, [-1, sen_len, label_num])
	lld = tf.nn.sigmoid_cross_entropy_with_logits(self.unary_score, plabel)
	self.targets = tf.argmax(self.unary_score, axis = 1)
	#self.cal = tf.reduce_sum(tf.cast(tf.equal(self.targets, tf.cast(self.label, tf.int64)), tf.int64))
        self.loss = tf.reduce_mean(lld)
        
        self.trains = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices = 1)
        length = tf.cast(length, tf.int32)
        return length

