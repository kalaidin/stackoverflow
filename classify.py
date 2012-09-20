#/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Features
"""
from time import time
import math
import getpass

import pandas
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.cross_validation import StratifiedKFold

DATA = ''
if getpass.getuser()=='marat':
    DATA = '/home/marat/kaggle.com/stackoverflow-data/'

def get_dataframe(filename):
    dataframe = pandas.io.parsers.read_csv(filename)
    return dataframe

def split_dataframe(df):
    kf = StratifiedKFold(df["OpenStatus"].values,5)
    train, test = kf.__iter__().next()
    return df.take(train), df.take(test)

def save_dataframe(df, filename):
    pass


input_features = [
    "PostId",
    "PostCreationDate",
    "OwnerUserId",
    "OwnerCreationDate",
    "ReputationAtPostCreation",
    "OwnerUndeletedAnswerCountAtPostTime",
    "Title",
    "BodyMarkdown",
    "Tag1",
    "Tag2",
    "Tag3",
    "Tag4",
    "Tag5",
    "PostClosedDate",
    "OpenStatus",
]

statuses = {
    "open": 0,
    "not a real question": 1,
    "not constructive": 2,
    "off topic": 3,
    "too localized": 4
}


def status_to_number(status):
    return statuses[status]

def extract_features(features, df):
    ff = pandas.DataFrame(index=df.index)
    for feature in features:
        if feature in df:
            if feature == "OpenStatus":
                ff = ff.join(df[feature].apply(status_to_number))
            else:
                ff = ff.join(df[feature])
    return ff


def mcll(predicted, actual):
    '''Calculate MCLL
    
    predicted -- numpy array(NxM) of probabilites where N -- num of obs and M is number of classes
    
    actual -- iterable integer 1-d array of actual classes'''      
    predicted = predicted/predicted.sum(1)[:,np.newaxis] #normalize
    return -np.sum(np.log(predicted[np.arange(predicted.shape[0]),actual]))/predicted.shape[0]

def calculate_mcll(probs, observations):
    mcll = 0.0
    for i, observation in enumerate(observations):
        mcll = mcll + math.log(probs[i][observation - 1])
    mcll = - mcll / len(observations)
    return mcll


#although it is recommended to put everything into main() 
# I prefer dirty way. Lots of benefits when using interactive
# IDE such as IPython
#if __name__ == "__main__":

print("Reading data...")

test_df = get_dataframe(DATA + "./train-sample.csv")
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

linear_decisions = clf.decision_function(test_tfidf_table)
predictedProbs = (1/(1+np.exp(-linear_decisions)))**3.5
print("MLCC: %f" %(mcll(predictedProbs, ff_val["OpenStatus"].values)))

t2 = time()
print("done in %d seconds" % (t2 - t1))
