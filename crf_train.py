from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from data import get_vocab, load_train_data_pipe, data_iter_crf
from model_crf import CRF

sen_len = 40
label_num = 5
sparse_len = 140
crf_num = 10720
learning_rate = 0.01
batch_size = 10
num_epoch = 10
dropout = True

print('read vocab ...')
vocab_size, embedding_size, embedding, vocab_w2i = get_vocab()
num_hidden = embedding_size

label_onehot = {}
label_hotone = {}

label_onehot['b-person'] = 1
label_onehot['i-person'] = 2
label_onehot['b-organization'] = 3
label_onehot['i-organization'] = 4
label_onehot['o'] = 0
label_hotone[1] = 'b-person'
label_hotone[2] = 'i-person'
label_hotone[3] = 'b-organization'
label_hotone[4] = 'i-organization'
#label_hotone[5] = 'o'
label_hotone[0] = 'o'
label_m = np.array([[0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 1], 
        [0, 0, 0, 1, 0], 
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0]])
files = []
dirs = '../OpenTargetedSentiment/data/pipe_en/'
for i in range(10):
    files.append('train'+str(i+1)+'.nn.ner')

files_t = []
for i in range(10):
    files_t.append('test'+str(i+1)+'.nn.ner')

for i in range(10):
    print('file ' + str(i+1))
    fw = open(dirs + files_t[i] + '.pipe_crf6', 'w')
    print('load train')
    doc, label, sparse, trans_tr = load_train_data_pipe(dirs+files[i], vocab_w2i, sen_len, sparse_len, crf_num, label_num, label_onehot)
    print('load test')
    doc_t, label_t, sparse_t, trans_t = load_train_data_pipe(dirs+files_t[i], vocab_w2i, sen_len, sparse_len, crf_num, label_num, label_onehot)
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            crf = CRF(sen_len, label_num, sparse_len, crf_num, learning_rate, label_m, trans_tr)
            sess.run(tf.initialize_all_variables())
            def train_step(input_, label_):
                feed_dict = {
                    crf.input : input_,
                    crf.label : label_
                }
                _, lss = sess.run([crf.trains, crf.loss], feed_dict)
            def test_step(input_, label_, fw, trans):
                totals_ = 0
                corrects_ = 0
                feed_dict = {crf.input : input_, crf.label : label_}
                unary_score, lens, _ = sess.run([crf.unary_score, crf.lens, crf.trains], feed_dict)
                for unary_, l_, lens_ in zip(unary_score, label_, lens):
                    u = unary_[:lens_]
                    l = l_[:lens_]
                    viterbi, _ = tf.contrib.crf.viterbi_decode(u, trans)
                    for k in range(lens_):
                        fw.write(label_hotone[l[k]]+' '+label_hotone[viterbi[k]]+'\n')
		    fw.write('\n')
                    corrects_ += np.sum(np.equal(viterbi, l))
                    totals_ += lens_
                return corrects_, totals_
            data = data_iter_crf(doc, label, sparse, num_epoch, batch_size)
            for input_, label_ in data:
	    	#print(label_)
                train_step(input_, label_)
            corrects = 0
            totals = 0
            test_data = data_iter_crf(doc_t, label_t, sparse_t, 1, 1)
	    trans = sess.run(crf.trans)
            for input_, label_ in test_data:
	    	#print(label_)
                corrects_, totals_ = test_step(input_, label_, fw, trans)
                corrects += corrects_
                totals += totals_
            print('accurry: ' + str(1.0 * corrects / totals))
            fw.close()

