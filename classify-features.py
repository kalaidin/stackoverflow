#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import cPickle
from time import time

from common import *

features = [
    "ReputationAtPostCreation",
    "OwnerUndeletedAnswerCountAtPostTime",
    "NumberOfTags",
    "BodyLength",
    "NumberOfWordsInTitle",
    "Age",
    "NumberOfWordsInBodymarkdown",
    "NumberOfCodeBlocksInBodymarkdown",
    "IsCodeSuppliedInBodymarkdown",
    "ProportionOfCodeToBodymarkdown"
]

#train_df = get_dataframe(DATA_PATH + "./train-sample.csv")
#train_ff = extract_features(input_features, train_df)
#all_labels = cPickle.load(open(DATA_PATH+"labels.numpy"))
#
#kf = StratifiedKFold(all_labels, 5)
#train, test = kf.__iter__().next()
#trainX, cvX = train_ff.take(train), train_ff.take(test)
#trainY, cvY = all_labels[train], all_labels[test]
#
#t = time()
#print("Training...")
#svc = RandomForestClassifier(n_estimators=4, verbose=2, compute_importances=False, n_jobs=-1)
#svc.fit(trainX, trainY)
#print( "%.2f seconds" %(time()-t) )
#
#test_probs = svc.predict_proba(cvX)
#print("MCLL on cv: %f" %mcll( cap_predictions(test_probs, 0.001) , cvY) )


# Creating submission...

train_df = get_dataframe(DATA_PATH + "./train-sample.csv")

train_x = extract_features(features, train_df)

#train_y = cPickle.load(open(DATA_PATH + "labels.numpy"))

print("Training...")
svc = RandomForestClassifier(n_estimators=400, compute_importances=False, n_jobs=-1)
svc.fit(train_x, train_df["OpenStatus"])

print("Predicting...")
test_df = get_dataframe(DATA_PATH + "./public_leaderboard.csv")
test_x = extract_features(features, test_df)
test_probs = svc.predict_proba(test_x)

updated_probs = update_probs(cap_predictions(test_probs, 0.001), get_train_sample_priors(), get_full_train_priors())

write_submission("submission_8.csv", updated_probs)