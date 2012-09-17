"""
Features
"""
import time
import math

import pandas
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm.sparse import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def get_dataframe(filename):
    dataframe = pandas.io.parsers.read_csv(filename)
    return dataframe


def split_dataframe(df, threshold=0.7):
    threshold_index = len(df) * threshold
    return df[:threshold_index], df[threshold_index:]


def save_dataframe(df, filename):
    pass


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
]

statuses = {
    "open": 1,
    "not a real question": 2,
    "not constructive": 3,
    "off topic": 4,
    "too localized": 5
}


def reduce_number_of_classes(status):
    return statuses[status]


def extract_features(features, df):
    ff = pandas.DataFrame(index=df.index)
    for feature in features:
        if feature in df:
            if feature == "OpenStatus":
                ff = ff.join(df[feature].apply(reduce_number_of_classes))
            else:
                ff = ff.join(df[feature])
    return ff


def filter_opened_questions(df):
    return df[df["OpenStatus"] == "open"]


def filter_closed_questions(df):
    return df[df["OpenStatus"] != "open"]


def calculate_mcll(probs, observations):
    mcll = 0.0
    for i, observation in enumerate(observations):
        mcll = mcll + math.log(probs[i][observation - 1])
    mcll = - mcll / len(observations)
    return mcll


if __name__ == "__main__":

    print("Reading data...")

    test_df = get_dataframe("./train-sample.csv")
    test_ff = extract_features(input_features, test_df)

    # split ~140k into ~100k training and ~40k test
    ff_train, ff_val = split_dataframe(test_ff)

    print("Training...")

    t1 = time.time()

    vectorizer = CountVectorizer()
    train_counts = vectorizer.fit_transform(ff_train["BodyMarkdown"])
    tfidf_transformer = TfidfTransformer()

    # 98190x285052
    train_tfidf_table = tfidf_transformer.fit_transform(train_counts)

    clf = LinearSVC().fit(train_tfidf_table, ff_train["OpenStatus"])

    print("Testing...")

    test_counts = vectorizer.transform(ff_val["BodyMarkdown"])
    test_tfidf_table = tfidf_transformer.transform(test_counts)

    predict = clf.predict(test_tfidf_table)
    print("np.mean: %f" % (np.mean(predict == ff_val["OpenStatus"])))

    t2 = time.time()
    print("done in %d seconds" % (t2 - t1))
