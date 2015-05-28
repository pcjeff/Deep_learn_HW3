
import numpy as np
import gensim
import time
import math
import theano
import theano.tensor as T
from collections import defaultdict

word_vec_len = 200 #dim of word vec
hidden_len = 400 #dim of hidden layer
output_len =  64402 #num of words, not sure yet
lr = 0.1
epoch = 10
sigma = lambda x: 1/ (1+T.exp(-x))
rng = np.random.RandomState(1234)
dtype = theano.config.floatX
v = T.matrix(dtype=dtype)
target = T.matrix(dtype=dtype)
params = []

def soft_max(x):
    deno = T.sum(T.exp(x))
    return x/deno

def init_w(size_x, size_y):
    values = np.array(rng.uniform(low = -1., high=1., size=(size_x, size_y)))
    return values

def get_param(word_vec_len, hidden_len, output_len):
    W_xh = theano.shared(init_w(hidden_len, word_vec_len))
    W_hy = theano.shared(init_w(output_len, hidden_len))
    W_hh = theano.shared(init_w(hidden_len, hidden_len))
    b_h = theano.shared(init_w(hidden_len, 1))
    b_y = theano.shared(init_w(output_len, 1))
    return [W_xh, W_hy, W_hh, b_y, b_h]

def one_rnn_step( x, h_tm, W_xh, W_hy, W_hh, b_y, b_h):
    #compute the output layer and the hidden layer of the RNN
    print 'W_xh: ' + str(W_xh.eval().shape)
    #print 'x: ' + str(x.eval().shape)
    print 'W_hh: ' + str(W_hh.eval().shape)
    #print 'h_tm: ' + str(h_tm.eval().shape)
    print 'b_h: ' + str(b_h.eval().shape)
    print 'W_hy: ' + str(W_hy.eval().shape)
    print 'b_y: ' + str(b_y.eval().shape)
    hh =  b_h
    #print 'hh' + str(hh.eval().shape)
    yy = sigma(theano.dot(W_hy, hh) + b_y)
    return [soft_max(yy), hh]

def get_word2vec_model(path='../word2vec-read-only/vectors.bin'):
    #read the vectors.bin into the word2vec model
    word2vec = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
    return word2vec

def get_train_func(cost, v, target):
    #get the learning function ==> learn_rnn_fn
    #output the cost, and update the params
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

def predict(line): # the function for jacky82226
    #return the probability of the sentence
    #(multiply all the prob of the words in the sentence)
    #param:
    #       type: list
    #       line: list of the words of the sentence  
    prob=1
    for i in range(line): 
        prob=prob*line[i]
    return prob
def main():
    
    [W_xh, W_hy, W_hh, b_y, b_h] = get_param(word_vec_len, hidden_len, output_len)
    params = [W_xh, W_hy, W_hh, b_y, b_h]
    h0 = theano.shared(np.zeros((hidden_len,1), dtype = dtype))
    # init the params 
    word2vec = get_word2vec_model()
    # the word2vec model
    index_word_mapping = dict(zip(word2vec.index2word, range(0, output_len)))
    # mapping of words and index EX: 1 for 'the', 2 for 'is'
    [y_vals, h_vals], _ = theano.scan(fn = one_rnn_step,
        truncate_gradient = 4,
        sequences = dict(input=v),
        outputs_info = [h0, None], #no output layer 
        non_sequences = params)
    cost = -T.mean(target * T.log(y_vals) + (1. - target) * T.log(1. - y_vals))
    learn_rnn_fn = get_train_func(cost, v, target)
    #above is for BPTT
    out = np.zeros((output_len,1), dtype=dtype)
    # for labeling the ground truth
    with open('train.txt', 'r') as f:
        for i in range(epoch):
            print 'train epoch ' + str(i) + str('...')
            for line in f:
                words = line.strip().split()
                for i in range(0, len(words)-1):
                    if words[i+1] in word2vec and words[i] in word2vec:
                        w_t = word2vec[words[i]].reshape(word_vec_len, 1)
                        out[index_word_mapping[words[i+1]]] = 1
                        learn_rnn_fn(w_t, out)
                        out[index_word_mapping[words[i+1]]] = 0

    return 0
if __name__ == '__main__':
    exit(main())
#    for line in f:
#        for word in line.split():
#            if word != '<s>' and word != '</s>':
#                word_tf[word] +=1
