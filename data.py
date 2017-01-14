#coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import math
#import sys

#reload(sys)
#sys.setdefaultencoding('utf-8')

def get_vocab():
    file1 = '../OpenTargetedSentiment/data/emd/embeddings_combine'
    f1 = open(file1)
    line = f1.readline()
    line = line.strip().split()
    if len(line) != 2:
        print('wrong ...')
        exit()
    vocab_size = int(line[0]) + 2
    embedding_size = int(line[1])
    print('vocab size: ' + str(vocab_size) + ' embedding size: ' + str(embedding_size))
    embedding = np.zeros(vocab_size * embedding_size).reshape(vocab_size, embedding_size).astype(np.float32)
    vocab_w2i = {}
    vocab_w2i['-padding_zero-'] = 0
    vocab_w2i['-unknown-'] = 1
    #not seen word, is 0. 
    i = 2
    for line in f1:
        line = line.strip().split()
        if len(line) != embedding_size+1:
            embedding[i] = np.array(line[:])
            vocab_w2i[i] = ' '
            i += 1
            continue
        embedding[i] = np.array(line[1:])
        vocab_w2i[line[0]] = i
        i += 1
    f1.close()
    return vocab_size, embedding_size, embedding, vocab_w2i
def get_trans(trans_m, tag_num):
    n = len(trans_m)
    trans = [[1.0 for l in range(tag_num)] for ll in range(tag_num)]
    for i in range(1, n):
        trans[trans_m[i]][trans_m[i-1]] += 1
    for i in range(tag_num):
        s = 0.0
        for j in range(tag_num):
            s += trans[i][j]
        for j in range(tag_num):
            trans[i][j] = math.log(trans[i][j] / s)
    return trans

def load_train_data_pipe(files, vocab_w2i, sen_len, sparse_len, crf_num, tag_num, label_onehot):
    fn = open(files)
    doc = []
    label = []
    sparse = []
    sen = [0] * sen_len
    #tag = [[0 for l in range(tag_num)] for ll in range(sen_len)]
    tag = [0] * sen_len
    #sp = [[0.0 for l in range(crf_num)] for ll in range(sen_len)]
    sp = [[0 for l in range(sparse_len)] for ll in range(sen_len)]
    trans_m = []
    i = 0
    for line in fn:
        line = line.strip().split()
        if len(line) < 1:
            doc.append(sen)
            label.append(tag)
            sparse.append(sp)
            i = 0
            sen = [0] * sen_len
            #tag = [[0 for l in range(tag_num)] for ll in range(sen_len)]
            tag = [0] * sen_len
            #sp = [0] * sparse_len
    	    sp = [[0.0 for l in range(sparse_len)] for ll in range(sen_len)]
            continue
        if line[0] == '-url-':
	    line[0] = 'url'
	elif line[0] == '-lrb-':
	    line[0] = 'lrb'
	elif line[0] == '-user-':
	    line[0] = 'user'
	elif line[0] == '-rrb-':
	    line[0] = 'rrb'

        if line[0].lower() in vocab_w2i:
            sen[i] = vocab_w2i[line[0].lower()]
        else:
            sen[i] = 1
        #tag.append(label_onehot[line[-1]])
        #tag[i][label_onehot[line[-1]]] = 1
        tag[i] = label_onehot[line[-1]]
        j = 0
        for a in line[1:-1]:
            #sp[i][int(a[3:])] = 1.0 #int(a[3:]) + crf_num * tag[i]
            sp[i][j] = int(a[3:])
            j += 1
	i += 1
        trans_m.append(int(label_onehot[line[-1]]))

    return doc, label, sparse, get_trans(trans_m, tag_num)

def load_train_data_joint(files, vocab_w2i, sen_len, sparse_len, crf_num, ner_num, sa_num, ner_onehot, sa_onehot):
    fn = open(files)
    doc = []
    ner = []
    sa = []
    sparse = []
    sen = [0] * sen_len
    #tag = [[0 for l in range(tag_num)] for ll in range(sen_len)]
    ner_ = [0] * sen_len
    sa_ = [0] * sen_len
    #sp = [[0.0 for l in range(crf_num)] for ll in range(sen_len)]
    sp = [[0 for l in range(sparse_len)] for ll in range(sen_len)]
    trans_ner = []
    trans_sa = []
    i = 0
    for line in fn:
        line = line.strip().split()
        if len(line) < 1:
            doc.append(sen)
            ner.append(ner_)
            sa.append(sa_)
            sparse.append(sp)
            i = 0
            sen = [0] * sen_len
            #tag = [[0 for l in range(tag_num)] for ll in range(sen_len)]
            ner_ = [0] * sen_len
            sa_  = [0] * sen_len
            #sp = [0] * sparse_len
    	    sp = [[0.0 for l in range(sparse_len)] for ll in range(sen_len)]
            continue
        if line[0] == '-url-':
	    line[0] = 'url'
	elif line[0] == '-lrb-':
	    line[0] = 'lrb'
	elif line[0] == '-user-':
	    line[0] = 'user'
	elif line[0] == '-rrb-':
	    line[0] = 'rrb'

        if line[0].lower() in vocab_w2i:
            sen[i] = vocab_w2i[line[0].lower()]
        else:
            sen[i] = 1
        #tag.append(label_onehot[line[-1]])
        #tag[i][label_onehot[line[-1]]] = 1
        ner_[i] = ner_onehot[line[-2]]
        sa_[i] = sa_onehot[line[-1]]
        j = 0
        for a in line[1:-2]:
            #sp[i][int(a[3:])] = 1.0 #int(a[3:]) + crf_num * tag[i]
            sp[i][j] = int(a[3:])
            j += 1
	i += 1
        trans_ner.append(int(ner_onehot[line[-2]]))
        trans_sa.append(int(sa_onehot[line[-1]]))
    return doc, ner, sa, sparse, get_trans(trans_ner, ner_num), get_trans(trans_sa, sa_num)


def data_iter(doc, label, sparse, num_epoch, batch_size):
    n = len(doc)
    doc = np.array(doc)
    label = np.array(label)
    #sparse = np.array(sparse)
    for k in range(num_epoch):
        i = 0
        while i + batch_size <= n:
            yield doc[i: i+batch_size], label[i: i+batch_size]
            i += batch_size
        if i < n:
            yield doc[i:], label[i:]

def data_iter_crf(doc, label, sparse, num_epoch, batch_size):
    n = len(doc)
    #doc = np.array(doc)
    label = np.array(label, dtype=np.int32)
    sparse = np.array(sparse)
    for k in range(num_epoch):
        i = 0
        while i + batch_size <= n:
            yield sparse[i: i+batch_size], label[i: i+batch_size]
            i += batch_size
        if i < n:
            yield sparse[i:], label[i:]

def data_iter_combine(doc, label, sparse, num_epoch, batch_size):
    n = len(doc)
    doc = np.array(doc)
    label = np.array(label)
    sparse = np.array(sparse)
    for k in range(num_epoch):
        i = 0
        while i + batch_size <= n:
            yield doc[i: i+batch_size], sparse[i:i+batch_size], label[i: i+batch_size]
            i += batch_size
        if i < n:
            yield doc[i:], sparse[i:], label[i:]

def data_iter_joint_dense(doc, ner, sa, num_epoch, batch_size):
    n = len(doc)
    doc = np.array(doc)
    ner = np.array(ner)
    sa  = np.array(sa)
    #sparse = np.array(sparse)
    for k in range(num_epoch):
        i = 0
        while i + batch_size <= n:
            yield doc[i: i+batch_size], ner[i:i+batch_size], sa[i: i+batch_size]
            i += batch_size
        if i < n:
            yield doc[i:], ner[i:], sa[i:]
def data_iter_joint_combine(doc, ner, sa, sparse, num_epoch, batch_size):
    n = len(doc)
    doc = np.array(doc)
    ner = np.array(ner)
    sa  = np.array(sa)
    sparse = np.array(sparse)
    for k in range(num_epoch):
        i = 0
        while i + batch_size <= n:
            yield doc[i: i+batch_size], ner[i:i+batch_size], sa[i: i+batch_size], sparse[i: i+batch_size]
            i += batch_size
        if i < n:
            yield doc[i:], ner[i:], sa[i:], sparse[i:]

