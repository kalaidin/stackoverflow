"""
Model to submit.
"""
import time

from sklearn.ensemble import RandomForestClassifier

from common import *

selected_features = [
    "ReputationAtPostCreation",
    "OwnerUndeletedAnswerCountAtPostTime",
    "NumberOfTags",
    "BodyLength",
    "Age",
    "NumberOfCodeBlocksInBodymarkdown",
    "ProportionOfCodeToBodymarkdown",
    "TitleLength"
]

if __name__ == "__main__":

    t_start = time.time()

    print("Reading training data...")
    train_parser = get_parser(DATA_PATH + "./train-sample.csv")
    train_x = extract_features(selected_features, train_parser)

    parser = get_parser(DATA_PATH + "./train-sample.csv")
    train_labels = extract_features(["OpenStatus"], parser, 'int32')

    print("Training classifier...")

    classifier = RandomForestClassifier(n_estimators=400,
        max_depth=12,
        compute_importances=False,
        n_jobs=-1)

    classifier.fit(train_x, train_labels.T[0])

    print("Reading public leaderboard data...")
    test_parser = get_parser(DATA_PATH + "./public_leaderboard.csv")
    test_x = extract_features(selected_features, test_parser)

    print("Predicting...")
    test_probs = classifier.predict_proba(test_x)

    updated_probs = update_probs(cap_predictions(test_probs, 0.001),
        get_train_sample_priors(), get_full_train_priors())

    write_submission("submission_final.csv", updated_probs)

    print("Done in %d seconds!" % (time.time() - t_start))
