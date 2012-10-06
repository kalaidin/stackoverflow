#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 21:03:45 2012
"""

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import cPickle
from time import time
t = time()

from common import *

#===========================================================================
# Parameters
#===========================================================================

SVC_C = 0.001
GENERATE_SUBMISSION = False

#===========================================================================
# Cross Validation
#===========================================================================

print("Reading data...")
overall = cPickle.load(open(DATA_PATH+"overall_train.sparse")).tocsr()
all_labels = cPickle.load(open(DATA_PATH+"labels.numpy"))

kf = StratifiedKFold(all_labels, 5)
train, test = kf.__iter__().next()
trainX, cvX = overall[train,:], overall[test ,:]
trainY, cvY = all_labels[train], all_labels[test]

print("Training a classifier...")
svc = LinearSVC(C=0.12, penalty='l1', dual=False )
#svc = LinearSVC(C=SVC_C)
svc.fit(trainX, trainY)

print svc.transform(trainX).shape

linear_decisions = svc.decision_function(cvX)
predicted_probs = (1 / (1 + np.exp(- linear_decisions))) ** 4.2
print("MCLL (Cross Validation): %f" %mcll(predicted_probs, cvY) )
print( "%.2f seconds" %(time()-t) )

#===========================================================================
# Submission
#===========================================================================

if GENERATE_SUBMISSION:
    print("Training a classifier...")
    svc_submission = LinearSVC(C=SVC_C)
    svc_submission.fit(overall, all_labels)
    
    print("Creating submission...")
    testX = cPickle.load(open(DATA_PATH+"overall_test.sparse")).tocsr()
    linear_decisions = svc_submission.decision_function(testX)
    predicted_probs = (1 / (1 + np.exp(- linear_decisions))) ** 4.1
    updated_probs = update_probs(predicted_probs, get_train_sample_priors(), get_full_train_priors())
    write_submission("submission_12.csv", updated_probs)
    
    print("Done!")