#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common methods
"""

from collections import Counter
import csv
import dateutil
import getpass
import re

import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold

import common

DATA_PATH = ''
if getpass.getuser() == 'marat':
    DATA_PATH = '/home/marat/kaggle.com/stackoverflow-data/'
elif getpass.getuser() == 'tsundokum':
    DATA_PATH = '/home/tsundokum/kaggle.com/stackoverflow-data/'

df_converters = {"PostCreationDate": dateutil.parser.parse,
                 "OwnerCreationDate": dateutil.parser.parse}


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
    "TitlePlusBody",
]

statuses = {
    "open": 3,
    "not a real question": 0,
    "off topic": 2,
    "not constructive": 1,
    "too localized": 4
}


def number_of_words(body_text):
    words = 0
    for line in body_text.split('\n'):
        if line.startswith('    '):
            continue
        words += len(line.split(' '))
    return words


def is_code_supplied(body_text):
    for line in body_text.split('\n'):
        if line.startswith('    '):
            return 1
    return 0


def number_of_lines_of_code(body_text):
    lines_of_code = 0
    for line in body_text.split('\n'):
        if line.startswith('    '):
            lines_of_code += 1
    return lines_of_code


def number_of_code_blocks(body_text):
    in_code_block = False
    code_blocks = 0
    for line in body_text.split('\n'):
        if line.strip() == '':
            continue
        if in_code_block:
            if line.startswith('    '):
                continue
            else:
                in_code_block = False
        else:
            if line.startswith('    '):
                in_code_block = True
                code_blocks += 1
            else:
                continue
    return code_blocks


def proportion_of_code_to_all_words(body_text):
    lines_of_code = number_of_lines_of_code(body_text)
    words = number_of_words(body_text)
    return lines_of_code / (lines_of_code + words / 7.0)


def number_of_words_in_bodymarkdown(df):
    return pd.DataFrame.from_dict({"NumberOfWordsInBodymarkdown":
        df["BodyMarkdown"].apply(number_of_words)})


def is_code_supplied_in_bodymarkdown(df):
    return pd.DataFrame.from_dict({"IsCodeSuppliedInBodymarkdown":
        df["BodyMarkdown"].apply(is_code_supplied)})


def proportion_of_code_to_bodymarkdown(df):
    return pd.DataFrame.from_dict({"ProportionOfCodeToBodymarkdown":
        df["BodyMarkdown"].apply(proportion_of_code_to_all_words)})


def number_of_code_blocks_in_bodymarkdown(df):
    return pd.DataFrame.from_dict({"NumberOfCodeBlocksInBodymarkdown":
        df["BodyMarkdown"].apply(number_of_code_blocks)})


def camel_to_underscores(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def status_to_number(status):
    return statuses[status]


def title_plus_body(df):
    return pd.DataFrame.from_dict({"TitlePlusBody":
        df["Title"] + ". " + df["BodyMarkdown"]})


def number_of_tags(df):
    return pd.DataFrame.from_dict({"NumberOfTags": [sum(map(lambda x:
        pd.isnull(x), row)) for row in (df[["Tag%d" % d
        for d in range(1, 6)]].values)]})["NumberOfTags"]


def number_of_words_in_title(df):
    return df["Title"].apply(lambda x: len(x.split()))


def body_length(df):
    return df["BodyMarkdown"].apply(len)


def title_length(df):
    return pd.DataFrame.from_dict({"TitleLength":
        df["Title"].apply(len)})


def age(df):
    return pd.DataFrame.from_dict({"Age": (df["PostCreationDate"]
            - df["OwnerCreationDate"]).apply(lambda x: x.total_seconds())})



def get_parser(filename):
    parser = pd.io.parsers.read_csv(
        filename,
        iterator=True,
        chunksize=100000,
        converters=df_converters)
    return parser


def extract_features(features, parser):
    all_ff = pd.DataFrame()
    for df in enumerate(parser):
        ff = pd.DataFrame(index=df.index)      
        for name in features:
            if name in df:
                if name == "OpenStatus":
                    ff = ff.join(df[name].apply(status_to_number))
                else:
                    ff = ff.join(df[name])
            else:
                ff = ff.join(getattr(common, camel_to_underscores(name))(df))
        all_ff = all_ff.append(ff)
    return all_ff



def mcll(predicted, actual):
    """
    Calculate MCLL.
    predicted -- numpy array(NxM) of probabilites,
        where N -- num of obs and M is number of classes
    actual -- iterable integer 1-d array of actual classes
    """
    predicted = predicted / predicted.sum(1)[:, np.newaxis]  # normalize
    return \
        - np.sum(np.log(predicted[np.arange(predicted.shape[0]), actual])) / \
        predicted.shape[0]


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


def get_number_of_questions(file_name):
    return sum(1 for r in get_reader(file_name))


def get_full_train_priors():
    # print(get_priors("train.csv"))
    return [
            0.00913477057600471,
            0.004645859639795308,
            0.005200965546050945,
            0.9791913907850639,
            0.0018270134530850952
            ]


def get_train_sample_priors():
    # get_priors("train-sample.csv")
    return [
            0.21949498117942284,
            0.11163311280939889,
            0.12497148397399338,
            0.5,
            0.043900422037184895
            ]


def update_probs(probs, old_priors, new_priors):
    # http://www.mpia-hd.mpg.de/Gaia/publications/probcomb_TN.pdf (equation 12)
    old_priors = np.kron(np.ones((np.size(probs, 0), 1)), old_priors)
    new_priors = np.kron(np.ones((np.size(probs, 0), 1)), new_priors)
    updated_probs = probs * new_priors * (1 - old_priors) / \
        (old_priors * (1 - probs - new_priors) + probs * new_priors)
    return updated_probs


def write_submission(file_name, probs):
    writer = csv.writer(open(file_name, "w"), lineterminator="\n")
    writer.writerows(probs)


def cap_predictions(probs, epsilon=0.001):
    probs[probs > 1 - epsilon] = 1 - epsilon
    probs[probs < epsilon] = epsilon
    row_sums = probs.sum(axis=1)
    probs = probs / row_sums[:, np.newaxis]
    return probs


if __name__ == "__main__":
    parser = get_parser("train-sample.csv")
    ff = extract_features(["PostId"], parser)
    print ff
