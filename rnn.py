
import numpy as np
import gensim
import time
import theano
import theano.tensor as T
from collections import defaultdict

word_vec_len = 200
hidden_len = 400
output_len = 10000000 #num of words, not sure yet
sigma = lambda x: 1/ (1+T.exp(-x))

def one_rnn_step( W_xh, W_hy, W_uh, b_u, b_h, x, u):
    hh = T.dot(x, W_xh) + T.dot(u, W_uh) + b_u
    yy = T.dot(hh, W_hy) + b_h
    return sigma(yy)
def init_w(size_x, size_y):
    values = np.array(rng.uniform(low = -1., high=1., size=(size_x, size_y)))
    return values

W_xh = theano.shared(init_w(word_vec_len, hidden_len))
W_hy = theano.shared(init_w(hidden_len, output_len))
W_uh = theano.shared(init_w(hidden_len, hidden_len))
b_u = theano.shared(init_w(hidden_len, 1))
b_h = theano.shared(init_w(output_len, 1))
    
word2vec = gensim.models.Word2Vec.load_word2vec_format('../word2vec-read-only/vectors.bin', binary=True)
word_tf = defaultdict(int)
with open('train.txt', 'r') as f:
    for line in f:
        for word in line.split():
            if word != '<s>' and word != '</s>':
                word_tf[word] +=1

#print max(word_tf.items(), key=lambda x:x[1])
