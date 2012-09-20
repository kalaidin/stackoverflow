#/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Features
"""
from time import time

from sklearn.cross_validation import StratifiedKFold

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from common import *

def split_dataframe(df):
    kf = StratifiedKFold(df["OpenStatus"].values,5)
    train, test = kf.__iter__().next()
    return df.take(train), df.take(test)

#def calculate_mcll(probs, observations):
#    mcll = 0.0
#    for i, observation in enumerate(observations):
#        mcll = mcll + math.log(probs[i][observation - 1])
#    mcll = - mcll / len(observations)
#    return mcll


#although it is recommended to put everything into main() 
# I prefer dirty way. Lots of benefits when using interactive
# IDE such as IPython
#if __name__ == "__main__":

print("Reading data...")

test_df = get_dataframe(DATA_PATH + "./train-sample.csv")
test_ff = extract_features(input_features, test_df)

# split ~140k into ~100k training and ~40k test
ff_train, ff_val = split_dataframe(test_ff)

print("Training...")

t1 = time()

vectorizer = CountVectorizer()
train_counts = vectorizer.fit_transform(ff_train["BodyMarkdown"])
tfidf_transformer = TfidfTransformer()

# 98190x285052
train_tfidf_table = tfidf_transformer.fit_transform(train_counts)

clf = LinearSVC().fit(train_tfidf_table, ff_train["OpenStatus"])

print("Testing...")

test_counts = vectorizer.transform(ff_val["BodyMarkdown"])
test_tfidf_table = tfidf_transformer.transform(test_counts)

predict = clf.predict(test_tfidf_table)
print("np.mean: %f" % (np.mean(predict == ff_val["OpenStatus"])))

#it is not very different from Uniform Benchmark =))
linear_decisions = clf.decision_function(test_tfidf_table)
predictedProbs = (1/(1+np.exp(-linear_decisions)))**3.5
print("MLCC: %f" %(mcll(predictedProbs, ff_val["OpenStatus"].values)))

t2 = time()
print("done in %d seconds" % (t2 - t1))
