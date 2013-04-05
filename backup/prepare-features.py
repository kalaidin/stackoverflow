#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 19:12:10 2012
"""

import cPickle
from common import *

def get_data():
    train_df = get_dataframe(DATA_PATH + "./train.csv")
    return extract_features(input_features, train_df), extract_features(["OpenStatus"], train_df)
    

train_ff, train_labels = get_data()

cPickle.dump(train_ff, open(DATA_PATH+"train-features.pandas","wb"),protocol=-1)
cPickle.dump(train_labels, open(DATA_PATH+"train-labels.pandas","wb"),protocol=-1)


