#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 21:03:45 2012
"""

from sklearn.svm import LinearSVC
import cPickle

from common import *

overall = cPickle.load(open(DATA_PATH+"overall_table.sparse")).tocsr()
all_labels = cPickle.load(open(DATA_PATH+"labels.numpy"))

kf = StratifiedKFold(labels, 5)
train, test = kf.__iter__().next()
trainX, cvX = overall[train,:], overall[test ,:]
trainY, cvY = all_labels[train], all_labels[test]

svc = LinearSVC(C=0.001)
svc.fit(trainX, trainY)

linear_decisions = svc.decision_function(cvX)
predicted_probs = (1 / (1 + np.exp(- linear_decisions))) ** 4
print("MCLL: %f" %mcll(predicted_probs, cvY) )

