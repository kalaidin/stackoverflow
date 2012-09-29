#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 20:54:43 2012
"""

from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from common import *

print("Reading data...")

test_df = get_dataframe(DATA_PATH + "./train-sample.csv")
test_ff = extract_features(input_features, test_df)


print("Training...")

t1 = time()

vectorizer = CountVectorizer()
train_counts = vectorizer.fit_transform(test_ff["TitlePlusBody"])
tfidf_transformer = TfidfTransformer(use_idf=False)

# 98190x285052
train_tfidf_table = tfidf_transformer.fit_transform(train_counts)

import cPickle
cPickle.dump(train_tfidf_table, open(DATA_PATH+"small_tables/title_body_logTF.sparse","wb"), protocol=-1)