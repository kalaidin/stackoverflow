#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 22:13:39 2012
"""

from sklearn.ensemble import RandomForestClassifier
import cPickle

from common import *

#===========================================================================
#Cross Validation
#===========================================================================

#from time import time
#
#t = time()
#print("Reading training data...")
#train_x = cPickle.load(open(DATA_PATH + "small_tables/train.numpy"))
#train_y = cPickle.load(open(DATA_PATH + "small_tables/train_labels.numpy")).T[0]
#
#kf = StratifiedKFold(train_y, 5)
#train, test = kf.__iter__().next()
#trainX, cvX = train_x[train,:], train_x[test,:]
#trainY, cvY = train_y[train], train_y[test]
#
#t = time()
#print("Training...")
#svc = RandomForestClassifier(n_estimators=100, verbose=2, max_depth=12, compute_importances=True, n_jobs=-1)
#svc.fit(trainX, trainY)
#print( "%.2f seconds" %(time()-t) )
#
#test_probs = svc.predict_proba(cvX)
#print("MCLL on cv: %f" %mcll( cap_predictions(test_probs, 0.001) , cvY) )
#
#print svc.feature_importances_

#===========================================================================
# Create Submition
#===========================================================================

print("Reading training data...")
train_x = cPickle.load(open(DATA_PATH + "small_tables/train.numpy"))
train_y = cPickle.load(open(DATA_PATH + "small_tables/train_labels.numpy")).T[0]

print("Training classifier...")
classifier = RandomForestClassifier(n_estimators=100, max_depth=12, verbose=2, compute_importances=False, n_jobs=3)
classifier.fit(train_x, train_y)

print("Reading public leaderboard data...")
test_x = cPickle.load(open(DATA_PATH + "small_tables/test.numpy"))

print("Predicting...")
test_probs = classifier.predict_proba(test_x)

#updated_probs = update_probs(cap_predictions(test_probs, 0.001), get_train_sample_priors(), get_full_train_priors())

write_submission("submission_14.csv", test_probs)

print("Done!")