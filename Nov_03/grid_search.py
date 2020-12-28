#!/usr/bin/env python3

# Team:
# 2f67b427-a885-11e7-a937-00505601122b
# b030d249-e9cb-11e9-9ce9-00505601122b
# 3351ff04-3f62-11e9-b0fd-00505601122b

import argparse
import sys
import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.7, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.


"""
https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
"""
def main(args):
    # Load digit dataset
    dataset = sklearn.datasets.load_digits()
    dataset.target = dataset.target % 2
    X = np.array(dataset.data)
    Y = np.array(dataset.target)

    # If you want to learn about the dataset, uncomment the following line.
    # print(dataset.DESCR)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_size,
                                                        random_state=args.seed)


    print(X_train.shape)
    # TODO: Create a pipeline, which
    minmax = MinMaxScaler()
    poly = PolynomialFeatures()
    clf = LogisticRegression(random_state=args.seed)
    pipeline = sklearn.pipeline.Pipeline([('minmax', minmax), ('poly', poly), ('clf', clf)])
    parameters = {'poly__degree': [1, 2], 'clf__C': [0.01, 1, 100], 'clf__solver': ('lbfgs', 'sag')}


    #skf = StratifiedKFold(2)
    #skf.get_n_splits(X_train, Y_train)
    #for train_index, test_index in skf.split(X_train, Y_train):
    #    X_train_, X_test_ = X_train[train_index], X_train[test_index]
    #    y_train_, y_test_ = Y_train[train_index], Y_train[test_index]
    search = sklearn.model_selection.GridSearchCV(estimator=pipeline, param_grid=parameters)
    search.fit(X_train, Y_train)
    #    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    #     print(search.best_params_)

    pred = search.predict(X_test)
    score = sklearn.metrics.accuracy_score(Y_test, pred)
    #print(score)

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
    test_accuracy = score

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)
    print("Test accuracy: {:.2f}".format(100 * test_accuracy))