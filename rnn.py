
import numpy as np
import gensim
import time
import theano
import theano.tensor as T
from collections import defaultdict

word_vec_len = 200 #dim of word vec
hidden_len = 4096 #dim of hidden layer
output_len =  64402 #num of words, not sure yet
lr = 0.1
sigma = lambda x: 1/ (1+T.exp(-x))
rng = np.random.RandomState(1234)
v = T.matrix(dtype=theano.config.floatX)
target = T.matrix(dtype=theano.config.floatX)


def init_w(size_x, size_y):
    values = np.array(rng.uniform(low = -1., high=1., size=(size_x, size_y)))
    return values

def get_param(word_vec_len, hidden_len, output_len):
    W_xh = theano.shared(init_w(word_vec_len, hidden_len))
    W_hy = theano.shared(init_w(hidden_len, output_len))
    W_hh = theano.shared(init_w(hidden_len, hidden_len))
    b_y = theano.shared(init_w(hidden_len, 1))
    b_h = theano.shared(init_w(output_len, 1))
    h0 = theano.shared(np.zeros(hidden_len, dtype = theano.config.floatX))
    return [W_xh, W_hy, W_hh, b_y, b_h]

def one_rnn_step( x, h_tm, W_xh, W_hy, W_hh, b_y, b_h):
    hh = T.dot(x, W_xh) + T.dot(h_tm, W_hh) + b_h
    yy = sigma(T.dot(hh, W_hy) + b_y)
    return [yy, hh]

def get_word2vec_model(path='../word2vec-read-only/vectors.bin'):
    word2vec = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
    return word2vec

def get_train_func(cost, v, target, params):
    gparams = []
    for param in params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    updates=[]
    for param, gparam in zip(params, gparams):
        updates.append((param, param - gparam * lr))
    learn_rnn_fn = theano.function(inputs = [v, target],
            outputs = cost,
            updates = updates
            )
    return learn_rnn_fn

def main():
    
    [params, h0] = get_param(word_vec_len, hidden_len, output_len, learning_rate)
    word2vec = get_word2vec_model()

    [y_vals, h_vals], _ = thenao.scan(fn = one_rnn_step,
        sequences = v,
        outputs_info = [h0, None], #no output layer 
        non_sequences = [W_xh, W_hy, W_hh, b_y, b_h],
        truncate_gradient = 4)
    cost = -T.mean(target * T.log(y_vals) + (1. - target) * T.log(1. - y_vals))
    learn_rnn_fn = get_train_func(cost, v, target, params)

    with open('train.txt', 'r') as f:
        for line in f:
            words = line.strip().split()
            for i in range(0, len(line)-2):
            w_t = word2vec[line[i]]
            #target = word2vec[line[i+1]]
            learn_rnn_fn(w_t, target)

#    for line in f:
#        for word in line.split():
#            if word != '<s>' and word != '</s>':
#                word_tf[word] +=1
