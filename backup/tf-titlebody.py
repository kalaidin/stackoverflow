#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 20:54:43 2012
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from common import *
import cPickle

print("Reading data...")

train_df = get_dataframe(DATA_PATH + "./train-sample.csv")
train_ff = extract_features(["TitlePlusBody"], train_df)

vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer(use_idf=False)

train_counts = vectorizer.fit_transform(train_ff["TitlePlusBody"])
train_ifidf = tfidf_transformer.fit_transform(train_counts)

cPickle.dump(train_ifidf, open(DATA_PATH+"small_tables/train_title_body_logTF.sparse","wb"), protocol=-1)

test_df = get_dataframe(DATA_PATH + "./public_leaderboard.csv")
test_ff = extract_features(["TitlePlusBody"], test_df)

test_count = vectorizer.transform(test_ff["TitlePlusBody"])
test_ifidf = tfidf_transformer.transform(test_count)

cPickle.dump(test_ifidf, open(DATA_PATH+"small_tables/test_title_body_logTF.sparse","wb"), protocol=-1)