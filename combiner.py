#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 20:47:10 2012
"""

import cPickle
from scipy.sparse import hstack
from common import *

small_tables_path = DATA_PATH + "small_tables/"

sparse_files = [
"keyword_tf.sparse",
"title_body_logTF.sparse"
]

tables = []
for sp_file in sparse_files:
    tables.append( cPickle.load( open(small_tables_path + sp_file) ) )
    
overall_table = hstack(tables)

cPickle.dump(overall_table, open(DATA_PATH + "overall_table.sparse", "wb"), protocol=-1)


data_raw = get_dataframe(DATA_PATH + "./train-sample.csv")
labels_frame = extract_features(['OpenStatus'], data_raw)
labels = labels_frame.values.T[0]

cPickle.dump(labels, open(DATA_PATH + "labels.numpy", "wb"), protocol=-1)
