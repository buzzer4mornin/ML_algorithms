#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

# Team:
# 2f67b427-a885-11e7-a937-00505601122b
# b030d249-e9cb-11e9-9ce9-00505601122b
# 3351ff04-3f62-11e9-b0fd-00505601122b

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x)
                    if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()
    # The input data are in dataset.data, targets are in dataset.target.
    X = dataset.data
    Y = dataset.target
    # If you want to learn about the dataset, uncomment the following line.
    # print(dataset.DESCR)

    # TODO: Append a new feature to all input data, with value "1"
    X += np.ones(X.shape)
    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=args.test_size, random_state=args.seed)

    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).

    weights = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)),
                     np.dot(X_train.T, Y_train))
    # TODO: Predict target values on the test set
    Y_predicted = np.dot(X_test, weights)

    # TODO: Compute root mean square error on the test set predictions
    rmse = np.math.sqrt(
        np.dot(np.subtract(Y_predicted, Y_test), np.subtract(Y_predicted, Y_test)) / Y_test.shape[0])

    return rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
