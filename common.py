#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some common functions and variables

Created on Thu Sep 20 23:00:26 2012
"""

from collections import Counter
import csv
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
    "off topic": 2,
    "not constructive": 3,
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
                "TitlePlusBody": df["Title"] + ". " + df["BodyMarkdown"]}))
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


def get_reader(file_name):
    reader = csv.reader(open(file_name))
    header = reader.next()
    return reader


def get_priors(file_name):
    closed_reasons = [r[14] for r in get_reader(file_name)]
    closed_reason_counts = Counter(closed_reasons)
    print closed_reason_counts
    total = float(len(closed_reasons))
    reasons = sorted(closed_reason_counts.keys(), key=lambda x: statuses[x])
    priors = [closed_reason_counts[reason] / total for reason in reasons]
    return priors


def get_full_train_priors():
    return [0.9791913907850639,
            0.00913477057600471,
            0.005200965546050945,
            0.004645859639795308,
            0.0018270134530850952]


def get_train_sample_priors():
    return [0.5,
            0.21949498117942284,
            0.12497148397399338,
            0.11163311280939889,
            0.043900422037184895]


def update_probs(probs, old_priors, new_priors):
    old_priors = np.kron(np.ones((np.size(probs, 0), 1)), old_priors)
    new_priors = np.kron(np.ones((np.size(probs, 0), 1)), new_priors)
    updated_probs = probs * new_priors * (1 - old_priors) / (old_priors * (1 - probs - new_priors) + probs * new_priors)
    return updated_probs


def write_submission(file_name, probs):
    writer = csv.writer(open(file_name, "w"), lineterminator="\n")
    writer.writerows(probs)
