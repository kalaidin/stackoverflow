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

from common import *

overall = cPickle.load(open(DATA_PATH+"overall_train.sparse")).tocsr()
all_labels = cPickle.load(open(DATA_PATH+"labels.numpy"))

kf = StratifiedKFold(all_labels, 5)
train, test = kf.__iter__().next()
trainX, cvX = overall[train,:], overall[test ,:]
trainY, cvY = all_labels[train], all_labels[test]

t = time()
#svc = LinearSVC(C=0.05, penalty='l1', dual=False )
svc = LinearSVC(C=0.05)
svc.fit(trainX, trainY)

linear_decisions = svc.decision_function(cvX)
predicted_probs = (1 / (1 + np.exp(- linear_decisions))) ** 4.1
print("MCLL: %f" %mcll(predicted_probs, cvY) )
print( "%.2f seconds" %(time()-t) )

testX = cPickle.load(open(DATA_PATH+"overall_test.sparse")).tocsr()

linear_decisions = svc.decision_function(testX)
predicted_probs = (1 / (1 + np.exp(- linear_decisions))) ** 4.1

updated_probs = update_probs(predicted_probs, get_train_sample_priors(), get_full_train_priors())
write_submission(DATA_PATH + "submission_4.csv", updated_probs)


#small_train_X = svc.transform(trainX)
#small_cv_X = svc.transform(cvX)

#rfc = RandomForestClassifier(n_estimators=2, max_depth=10, min_samples_split=1, random_state=0)
#rfc.fit( small_train_X.todense(), trainY )   
#probs = rfc.predict_proba( small_cv_X.todense() )                         
#print("MCLL after random forest: %f" %mcll(probs, cvY) )


#gbc = GradientBoostingClassifier(n_estimators=2, learn_rate=1.0, max_depth=1, random_state=0)
#gbc.fit( small_train_X.todense(), trainY )   
#probs = gbc.predict_proba( small_cv_X.todense() )                         
#print("MCLL after GBC: %f" %mcll(probs, cvY) )
#


