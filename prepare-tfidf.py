#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 22:00:19 2012
"""
from common import *
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import dok_matrix, csr_matrix, lil_matrix
import cPickle

import nltk

TRAIN_FILE = DATA_PATH + "train.csv"
TEST_FILE = DATA_PATH + "public_leaderboard.csv"
STOP_WORDS = set(nltk.corpus.stopwords.words('english'))

vectorizer = CountVectorizer(stop_words="english")
tok = vectorizer.build_tokenizer()

def exclude_code(body_text):
    return '\n'.join([line for line in body_text.split('\n') if not line.startswith("    ")])

once_set = set()
words_set = set()

print "building words set"
for i, question in enumerate(get_reader(TRAIN_FILE)):
    title = question[6]
    body = exclude_code(question[7])
    #tags = [question[i] for i in [8,9,10,11,12] if type(question[i]) is str]
    words = tok('. '.join([title, body] ).lower())
    for word in (w for w in words if w not in STOP_WORDS):
        if word in once_set:
            words_set.add(word)
        else:
            once_set.add(word)
                
    if (i+1)%100000 == 0:
        print i,
#        break
print "done!"
    
indexi = dict(((v,k) for k,v in enumerate(words_set)))

print "building sparse representation"
tf = lil_matrix((i,len(indexi)), dtype='int32')

for i, question in enumerate(get_reader(TRAIN_FILE)):
    title = question[6]
    body = exclude_code(question[7])
    words = tok('. '.join([title, body] ).lower())
    for word in (w for w in words if (w not in STOP_WORDS) and (w in words_set)):
        index = indexi[word]
        tf[i, index] += 1
    if (i+1)%100000 == 0:
        print i,

print "Done!"
cPickle.dump(tf.tocsr(), open(DATA_PATH+"small_tables/train_titlebody_tf.sparse","wb"), protocol=-1)

print "building sparse representation for test data"
tf = lil_matrix((73290,len(indexi)), dtype='int32')
for i, question in enumerate(get_reader(TEST_FILE)):
    title = question[6]
    body = exclude_code(question[7])
    words = tok('. '.join([title, body] ).lower())
    for word in (w for w in words if w not in STOP_WORDS):
        if word in words_set:
            index = indexi[word]
            tf[i, index] += 1
print "done!"
cPickle.dump(tf.tocsr(), open(DATA_PATH+"small_tables/test_titlebody_tf.sparse","wb"), protocol=-1)