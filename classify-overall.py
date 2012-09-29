#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 21:03:45 2012
"""

from common import *

import cPickle

overall = cPickle.load(open(DATA_PATH+"overall_table.sparse"))

kf = StratifiedKFold(df["OpenStatus"].values, 5)
train, test = kf.__iter__().next()
trainX, cvX = overall[train,:], overall[test ,:]