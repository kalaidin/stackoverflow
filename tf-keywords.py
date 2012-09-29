#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:51:35 2012
"""

from common import *
import pandas


from sklearn.feature_extraction.text import CountVectorizer

def convert_to_string(row):
    strings = [str(keyword) for keyword in row if not pandas.isnull(keyword) ]
    return ' '.join(strings)
    

data_train = get_dataframe(DATA_PATH + "./train-sample.csv")
train = extract_features(['Tag%d' %i for i in range(1,6)], data_train)

data_test= get_dataframe(DATA_PATH + "./public_leaderboard.csv")
test = extract_features(['Tag%d' %i for i in range(1,6)], data_test)

keywords_train = train.apply( convert_to_string , axis=1 )
keywords_test = test.apply( convert_to_string , axis=1 )

vectorizer = CountVectorizer(stop_words=None,analyzer = 'word',token_pattern='[^\s]+')
counts_train = vectorizer.fit_transform(keywords_train.values)
counts_test = vectorizer.transform(keywords_test.values)

import cPickle
cPickle.dump(counts_train, open(DATA_PATH+"small_tables/train_keyword_tf.sparse","wb"), protocol=-1)
cPickle.dump(counts_test, open(DATA_PATH+"small_tables/test_keyword_tf.sparse","wb"), protocol=-1)