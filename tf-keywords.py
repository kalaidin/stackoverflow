#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:51:35 2012
"""

from common import *
import pandas

data_raw = get_dataframe(DATA_PATH + "./train-sample.csv")
data = extract_features(['Tag%d' %i for i in range(1,6)], data_raw)

def convert_to_string(row):
    strings = [str(keyword) for keyword in row if not pandas.isnull(keyword) ]
    return ' '.join(strings)

keywords_rows = data.apply( convert_to_string , axis=1 )

keywords_rows.to_csv(DATA_PATH + "/keywords.csv", header=False, index=False, mode='wb')

