#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 22:00:19 2012
"""
from common import *
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import dok_matrix

vectorizer = CountVectorizer(stop_words="english")
tok = vectorizer.build_tokenizer()

words_set = set()

file_name = DATA_PATH + "train.csv"
for i, question in enumerate(get_reader(file_name)):
    title = question[6]
    body =  question[7]
    #tags = [question[i] for i in [8,9,10,11,12] if type(question[i]) is str]
    words = tok('. '.join([title, body] ).lower())
    for word in words:
        if word not in words_set:
            words_set.add(word)
        
    if i>2:
        break
    
indexi = dict(((v,k) for k,v in enumerate(words_set)))
tf = dok_matrix((i,len(indexi)), dtype='int32')
for i, question in enumerate(get_reader(file_name)):
    title = question[6]
    body =  question[7]
    words = tok('. '.join([title, body] ).lower())
    for word in words:
        index = indexi[word]
        tf[i, index] += 1
        
    if i>2:
        break
 