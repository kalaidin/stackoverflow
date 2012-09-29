#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:51:35 2012
"""

from common import *
import pandas


from sklearn.feature_extraction.text import CountVectorizer

data_raw = get_dataframe(DATA_PATH + "./train-sample.csv")
data = extract_features(['Tag%d' %i for i in range(1,6)], data_raw)

def convert_to_string(row):
    strings = [str(keyword) for keyword in row if not pandas.isnull(keyword) ]
    return ' '.join(strings)
    

keywords_rows = data.apply( convert_to_string , axis=1 )

vectorizer = CountVectorizer(stop_words=None,analyzer = 'word',token_pattern='[^\s]+')
counts = vectorizer.fit_transform(keywords_rows.values)

import cPickle
cPickle.dump(counts, open(DATA_PATH+"small_tables/keyword_tf.sparse","wb"), protocol=-1)