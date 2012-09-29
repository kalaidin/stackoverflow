#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some common functions and variables

Created on Thu Sep 20 23:00:26 2012
"""

import getpass
import math

import pandas
import numpy as np
from sklearn.cross_validation import StratifiedKFold

DATA_PATH = ''
if getpass.getuser() == 'marat':
    DATA_PATH = '/home/marat/kaggle.com/stackoverflow-data/'

input_features = [
    "PostId",
    "PostCreationDate",
    "OwnerUserId",
    "OwnerCreationDate",
    "ReputationAtPostCreation",
    "OwnerUndeletedAnswerCountAtPostTime",
    "Title",
    "BodyMarkdown",
    "Tag1",
    "Tag2",
    "Tag3",
    "Tag4",
    "Tag5",
    "PostClosedDate",
    "OpenStatus",
    "TitlePlusBody"
]

statuses = {
    "open": 0,
    "not a real question": 1,
    "not constructive": 2,
    "off topic": 3,
    "too localized": 4
}


def get_dataframe(filename):
    dataframe = pandas.io.parsers.read_csv(filename)
    return dataframe


def status_to_number(status):
    return statuses[status]


def extract_features(features, df):
    ff = pandas.DataFrame(index=df.index)
    for feature in features:
        if feature in df:
            if feature == "OpenStatus":
                ff = ff.join(df[feature].apply(status_to_number))
            else:
                ff = ff.join(df[feature])
        elif feature == "TitlePlusBody":
            ff = ff.join(pandas.DataFrame.from_dict({
                "TitlePlusBody": df["Title"] + ". "+ df["BodyMarkdown"]}))
        else:
            pass
    return ff


def mcll(predicted, actual):
    """Calculate MCLL

    predicted -- numpy array(NxM) of probabilites,
        where N -- num of obs and M is number of classes

    actual -- iterable integer 1-d array of actual classes
    """
    predicted = predicted / predicted.sum(1)[:, np.newaxis]  # normalize
    return - np.sum(np.log(predicted[np.arange(predicted.shape[0]), actual])) / predicted.shape[0]

    
def split_dataframe(df):
    kf = StratifiedKFold(df["OpenStatus"].values, 5)
    train, test = kf.__iter__().next()
    return df.take(train), df.take(test)    
