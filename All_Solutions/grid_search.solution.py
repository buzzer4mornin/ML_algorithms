#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load digit dataset
    dataset = sklearn.datasets.load_digits()
    dataset.target = dataset.target % 2

    # If you want to learn about the dataset, uncomment the following line.
    # print(dataset.DESCR)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a pipeline, which
    # 1. performs sklearn.preprocessing.MinMaxScaler()
    # 2. performs sklearn.preprocessing.PolynomialFeatures()
    # 3. performs sklearn.linear_model.LogisticRegression(random_state=args.seed)
    #
    # Then, using sklearn.model_selection.StratifiedKFold(5), evaluate crossvalidated
    # train performance of all combinations of the the following parameters:
    # - polynomial degree: 1, 2
    # - LogisticRegression regularization C: 0.01, 1, 100
    # - LogisticRegression solver: lbfgs, sag
    #
    # For the best combination of parameters, compute the test set accuracy.
    #
    # The easiest way is to use `sklearn.model_selection.GridSearchCV`.
    model = sklearn.pipeline.Pipeline([
        ("scaling", sklearn.preprocessing.MinMaxScaler()),
        ("polynomial", sklearn.preprocessing.PolynomialFeatures()),
        ("lr", sklearn.linear_model.LogisticRegression(random_state=args.seed)),
    ])
    model = sklearn.model_selection.GridSearchCV(
        model,
        {"polynomial__degree": [1, 2],
         "lr__C": [0.01, 1, 100],
         "lr__solver": ["lbfgs", "sag"]},
        cv=sklearn.model_selection.StratifiedKFold(5),
    )
    model.fit(train_data, train_target)
    for rank, accuracy, params in zip(model.cv_results_["rank_test_score"], model.cv_results_["mean_test_score"], model.cv_results_["params"]):
        print("Rank: {:2d} Cross-val: {:.1f}%".format(rank, 100 * accuracy), *("{}: {:<5}".format(key, value) for key, value in params.items()))
    test_accuracy = sklearn.metrics.accuracy_score(test_target, model.predict(test_data))

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)
    print("Test accuracy: {:.2f}".format(100 * test_accuracy))