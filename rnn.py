
import numpy as np
import gensim
import time
import theano
from collections import defaultdict

word2vec = gensim.models.Word2Vec.load_word2vec_format('../word2vec-read-only/vectors.bin', binary=True)

word_tf = defaultdict(int)


with open('train.txt', 'r') as f:
    for line in f:
        for word in line.split():
            if word != '<s>' and word != '</s>':
                word_tf[word] +=1

#print max(word_tf.items(), key=lambda x:x[1])
