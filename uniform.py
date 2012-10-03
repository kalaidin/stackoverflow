#/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 00:52:46 2012
"""
from common import get_dataframe, extract_features, DATA_PATH
import numpy as np

data_raw = get_dataframe(DATA_PATH + "./train-sample.csv")

labels_frame = extract_features(['OpenStatus'], data_raw)
labels = labels_frame.values.T[0].view('int32')

counts = np.bincount(labels.view('int32'))
proportions = counts/float(counts.sum())

a = np.ones(test_probs.shape) * proportions[np.newaxis,:]

print("MCLL on cv: %f" %mcll( a, cvY ))