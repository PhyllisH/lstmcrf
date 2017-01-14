from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from data import get_vocab, load_train_data_joint, data_iter_joint_dense
from model_joint_dense import JOINT_DENSE

sen_len = 40
ner_num = 5
sa_num = 5
sparse_len = 140
crf_num = 10720
learning_rate = 0.01
batch_size = 10
num_epoch = 10
dropout = True

print('read vocab ...')
vocab_size, embedding_size, embedding, vocab_w2i = get_vocab()
num_hidden = embedding_size

ner_onehot = {}
ner_hotone = {}
ner_onehot['b-person'] = 1
ner_onehot['i-person'] = 2
ner_onehot['b-organization'] = 3
ner_onehot['i-organization'] = 4
ner_onehot['o'] = 0
ner_hotone[1] = 'b-person'
ner_hotone[2] = 'i-person'
ner_hotone[3] = 'b-organization'
ner_hotone[4] = 'i-organization'
ner_hotone[0] = 'o'
sa_onehot = {}
sa_hotone = {}
sa_onehot['b-positive'] = 1
sa_onehot['i-positive'] = 2
#sa_onehot['b-neutral'] = 3
#sa_onehot['i-neutral'] = 4
sa_onehot['b-negative'] = 3
sa_onehot['i-negative'] = 4
sa_onehot['o'] = 0
sa_hotone[1] = 'b-positive'
sa_hotone[2] = 'i-positive'
#sa_hotone[3] = 'b-neutral'
#sa_hotone[4] = 'i-neutral'
sa_hotone[3] = 'b-negative'
sa_hotone[4] = 'b-negative'
sa_hotone[0] = 'o'


files = []
dirs = '../OpenTargetedSentiment/data/2joint_en/'
for i in range(10):
    files.append('train'+str(i+1)+'.nn')

files_t = []
for i in range(10):
    files_t.append('test'+str(i+1)+'.nn')

for i in range(10):
    print('file ' + str(i+1))
    fw_ner = open(dirs + files_t[i] + '.joint.dense.ner1', 'w')
    fw_sa = open(dirs + files_t[i] + '.joint.dense.sa1', 'w')
    print('load train')
    doc, ner_, sa_, _, _, _ = load_train_data_joint(dirs+files[i], vocab_w2i, sen_len, sparse_len, crf_num, ner_num, sa_num, ner_onehot, sa_onehot)
    print('load test')
    doc_t, ner_t, sa_t, _, _, _ = load_train_data_joint(dirs+files_t[i], vocab_w2i, sen_len, sparse_len, crf_num, ner_num, sa_num, ner_onehot, sa_onehot)
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            lstm = JOINT_DENSE(sen_len, num_hidden, ner_num, sa_num, vocab_size, embedding_size, embedding, learning_rate)
            sess.run(tf.initialize_all_variables())
            def train_step(input_, ner_tr, sa_tr, d):
                feed_dict = {
                    lstm.input : input_,
                    lstm.ner : ner_tr,
                    lstm.sa  : sa_tr,
		    lstm.dropout: d
                }
                _, lss = sess.run([lstm.trains, lstm.loss], feed_dict)
            def test_step(input_, ner_t_, sa_t_, d, fw_ner, fw_sa, trans_ner, trans_sa):
                totals_ner_ = 0
                corrects_ner_ = 0
                totals_sa_ = 0
                corrects_sa_ = 0
                feed_dict = {lstm.input : input_, lstm.dropout : d}
                unary_ner_score, unary_sa_score, lens = sess.run([lstm.unary_ner_score, lstm.unary_sa_score, lstm.sen_lens], feed_dict)
                for unary_, l_, lens_ in zip(unary_ner_score, ner_t_, lens):
                    u = unary_[:lens_]
                    l = l_[:lens_]
                    viterbi, _ = tf.contrib.crf.viterbi_decode(u, trans_ner)
                    for k in range(lens_):
                        fw_ner.write(ner_hotone[l[k]]+' '+ner_hotone[viterbi[k]]+'\n')
		    fw_ner.write('\n')
                    corrects_ner_ += np.sum(np.equal(viterbi, l))
                    totals_ner_ += lens_

                for unary_, l_, lens_ in zip(unary_sa_score, sa_t_, lens):
                    u = unary_[:lens_]
                    l = l_[:lens_]
                    viterbi, _ = tf.contrib.crf.viterbi_decode(u, trans_sa)
                    for k in range(lens_):
                        fw_sa.write(sa_hotone[l[k]]+' '+sa_hotone[viterbi[k]]+'\n')
		    fw_sa.write('\n')
                    corrects_sa_ += np.sum(np.equal(viterbi, l))
                    totals_sa_ += lens_

                return corrects_ner_, totals_ner_, corrects_sa_, totals_sa_
            data = data_iter_joint_dense(doc, ner_, sa_, num_epoch, batch_size)
            for input_, ner_tr, sa_tr in data:
                train_step(input_, ner_tr, sa_tr, True)
            corrects_ner = 0
            totals_ner = 0
            corrects_sa = 0
            totals_sa = 0
            test_data = data_iter_joint_dense(doc_t, ner_t, sa_t, 1, 1)
	    trans_ner = sess.run(lstm.trans_ner)
            trans_sa = sess.run(lstm.trans_sa)
            for input_, ner_t_, sa_t_, in test_data:
                corrects_ner_, totals_ner_, corrects_sa_, totals_sa_ = test_step(input_, ner_t_, sa_t_, False, fw_ner, fw_sa, trans_ner, trans_sa)
                corrects_ner += corrects_ner_
                totals_ner += totals_ner_
                corrects_sa += corrects_sa_
                totals_sa += totals_sa_
            print('accurry ner: ' + str(1.0 * corrects_ner / totals_ner))
            print('accurry sa: ' + str(1.0 * corrects_sa / totals_sa))
            fw_ner.close()
            fw_sa.close()

