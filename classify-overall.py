#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 21:03:45 2012
"""

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import cPickle
from time import time

from common import *

overall = cPickle.load(open(DATA_PATH+"overall_table.sparse")).tocsr()
all_labels = cPickle.load(open(DATA_PATH+"labels.numpy"))

kf = StratifiedKFold(all_labels, 5)
train, test = kf.__iter__().next()
trainX, cvX = overall[train,:], overall[test ,:]
trainY, cvY = all_labels[train], all_labels[test]

t = time()
svc = LinearSVC(C=0.05, penalty='l1', dual=False )
svc.fit(trainX, trainY)

small_train_X = svc.transform(trainX)
small_cv_X = svc.transform(cvX)

#linear_decisions = svc.decision_function(cvX)
#predicted_probs = (1 / (1 + np.exp(- linear_decisions))) ** 4.1
#print("MCLL: %f" %mcll(predicted_probs, cvY) )
#print( "%.2f seconds" %(time()-t) )

rfc = RandomForestClassifier(n_estimators=10, max_depth=10, min_samples_split=1, random_state=0)
rfc.fit( small_train_X.todense(), trainY )   
probs = rfc.predict_proba( small_cv_X )                         
print("MCLL after random forest: %f" %mcll(probs, cvY) )



