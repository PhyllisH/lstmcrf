import tensorflow as tf

class CRF():
    def __init__(self, sen_len, label_num, sparse_num, crf_num, learning_rate, label_m, trans):
        self.input = tf.placeholder(tf.int32, [None, sen_len, sparse_num], name = 'input')
        self.label = tf.placeholder(tf.int32, [None, sen_len], name = 'label')

        #param = tf.get_variable('param', shape = [crf_num], initializer = tf.truncated_normal_initializer(stddev=0.01))
        #pa = tf.get_variable('pa', shape = [crf_num, label_num], initializer = tf.truncated_normal_initializer(stddev=0.01))
        pa = tf.get_variable('pa', shape = [crf_num, label_num], initializer = tf.truncated_normal_initializer(stddev=0.01))
        #self.trans = tf.Variable(trans, 'trans')
        self.trans = tf.Variable(tf.zeros([label_num, label_num]), name = 'label_trans')
        #vec = tf.reshape(tf.nn.embedding_lookup(param, self.input), [-1, crf_num])
        vec = tf.nn.embedding_lookup(pa, self.input)
        mask = tf.reshape(tf.sign(self.input), [-1])
        vec = tf.reshape(tf.transpose(tf.transpose(tf.reshape(vec, [-1, label_num])) * tf.cast(mask, tf.float32)), [-1, sen_len, sparse_num, label_num])
        vec_ = tf.reduce_mean(vec, 2)
	self.unary_score = tf.reshape(vec_, [-1, sen_len, label_num])
     
        self.lens = self.length(self.label)
        lld, self.trans = tf.contrib.crf.crf_log_likelihood(self.unary_score, self.label, self.lens, self.trans)
        self.loss = tf.reduce_mean(-lld)
        self.trains = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        #self.trains = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices = 1)
        length = tf.cast(length, tf.int32)
        return length

