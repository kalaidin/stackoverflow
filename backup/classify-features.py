#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
import cPickle

from common import *

features = [
    "ReputationAtPostCreation",
    "OwnerUndeletedAnswerCountAtPostTime",
    "NumberOfTags",
    "BodyLength",
#    "NumberOfWordsInTitle",
    "Age",
#    "NumberOfWordsInBodymarkdown",
    "NumberOfCodeBlocksInBodymarkdown",
#    "IsCodeSuppliedInBodymarkdown",
    "ProportionOfCodeToBodymarkdown",
    "TitleLength"    
]

#===========================================================================
#Cross Validation
#===========================================================================

from time import time

t = time()
print("Reading training data...")
train_parser = get_dataframe(DATA_PATH + "train.csv")
train_ff = extract_features(features, train_parser)
all_labels = extract_features(["OpenStatus"], train_parser)
print("%.3f sec" % (time()-t))



print("Training classifier...")
kf = StratifiedKFold(all_labels, 5)
train, test = kf.__iter__().next()
trainX, cvX = train_ff.take(train), train_ff.take(test)
trainY, cvY = all_labels[train], all_labels[test]

t = time()
print("Training...")
svc = RandomForestClassifier(n_estimators=4, verbose=2, max_depth=12, compute_importances=False, n_jobs=-1)
svc.fit(trainX, trainY)
print( "%.2f seconds" %(time()-t) )

test_probs = svc.predict_proba(cvX)
print("MCLL on cv: %f" %mcll( cap_predictions(test_probs, 0.001) , cvY) )


#===========================================================================
# Create Submition
#===========================================================================

#print("Reading training data...")
#train_df = get_dataframe(DATA_PATH + "./train-sample.csv")
#train_x = extract_features(features, train_df)
#
#print("Training classifier...")
#classifier = RandomForestClassifier(n_estimators=400, compute_importances=False, n_jobs=-1)
#classifier.fit(train_x, train_df["OpenStatus"])
#
#print("Reading public leaderboard data...")
#test_df = get_dataframe(DATA_PATH + "./public_leaderboard.csv")
#test_x = extract_features(features, test_df)
#
#print("Predicting...")
#test_probs = classifier.predict_proba(test_x)
#
#updated_probs = update_probs(cap_predictions(test_probs, 0.001), get_train_sample_priors(), get_full_train_priors())
#
#write_submission("submission_9.csv", updated_probs)
#
#print("Done!")