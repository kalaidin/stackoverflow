#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:51:35 2012
"""

from common import *
import pandas

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

data_raw = get_dataframe(DATA_PATH + "./train-sample.csv")
data = extract_features(['Tag%d' %i for i in range(1,6)], data_raw)

def convert_to_string(row):
    strings = [str(keyword) for keyword in row if not pandas.isnull(keyword) ]
    return ' '.join(strings)
    

keywords_rows = data.apply( convert_to_string , axis=1 )

#keywords_rows.to_csv(DATA_PATH + "/keywords.csv", header=False, index=False, mode='wb')

#vectorizer = CountVectorizer(stop_words=None)
#train_counts = vectorizer.fit_transform(keywords_rows.values)
#
#tfidf_transformer = TfidfTransformer(norm=None, use_idf=False)
#train_tfidf_table = tfidf_transformer.fit_transform(train_counts)
#
#
#clf = LinearSVC().fit(train_tfidf_table, ff_train["OpenStatus"])
#
#print("Testing...")
#
#test_counts = vectorizer.transform(ff_val["BodyMarkdown"])
#test_tfidf_table = tfidf_transformer.transform(test_counts)
#
#predict = clf.predict(test_tfidf_table)
#print("np.mean: %f" % (np.mean(predict == ff_val["OpenStatus"])))
#
##it is not very different from Uniform Benchmark =))
#linear_decisions = clf.decision_function(test_tfidf_table)
#predictedProbs = (1/(1+np.exp(-linear_decisions)))**3.5
#print("MLCC: %f" %(mcll(predictedProbs, ff_val["OpenStatus"].values)))
#
#t2 = time()
#print("done in %d seconds" % (t2 - t1))
#

