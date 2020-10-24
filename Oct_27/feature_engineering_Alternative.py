#!/usr/bin/env python3

# Team:
# 2f67b427-a885-11e7-a937-00505601122b
# b030d249-e9cb-11e9-9ce9-00505601122b
# 3351ff04-3f62-11e9-b0fd-00505601122b


import argparse
import numpy as np
import pandas as pd
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="boston", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()

    X = np.array(dataset.data)
    Y = np.array(dataset.target).reshape(-1, 1)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_size,
                                                        random_state=args.seed)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    # TODO: Process the input columns in the following way:
    #
    # - if a column has only integer values, consider it a categorical column
    #   (days in a week, dog breed, ...; in general integer values can also
    #   represent numerical non-categorical values, but we use this assumption
    #   for the sake of an exercise). Encode the values with one-hot encoding
    #   using `sklearn.preprocessing.OneHotEncoder` (note that its output is by
    #   default sparse, you can use `sparse=False` to generate dense output;
    #   also use `handle_unknown="ignore"` to ignore missing values in test set).
    #
    # - for the rest of the columns, normalize their values so that they
    #   have mean 0 and variance 1; use `sklearn.preprocessing.StandardScaler`.
    #
    # In the output, there should be first all the one-hot categorical features,
    # and then the real-valued features. To process different dataset columns
    # differently, you can use `sklearn.compose.ColumnTransformer`.

    try:
        # Check categorical columns
        categ_check = np.all(X.astype(int) == X, axis=0)
        categ_colnames = [i for i, x in enumerate(categ_check) if x]

        #print(X_train.shape,X_test.shape)

        # Initialize Encoder
        myencoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

        # fit Encoder to separate columns
        encoded_cols = [np.c_[myencoder.fit_transform(np.array(X_train.loc[:, i]).reshape(-1, 1))] for i in
                        categ_colnames]
        encoded_cols = np.c_[tuple(encoded_cols)]
        encoded_cols_1 = [np.c_[myencoder.fit_transform(np.array(X_test.loc[:, i]).reshape(-1, 1))] for i in
                        categ_colnames]                                              #yyyyyyy
        encoded_cols_1 = np.c_[tuple(encoded_cols_1)]                                 #yyyyyyy

        #print(encoded_cols.shape, encoded_cols_1.shape)

        # drop Encoded columns
        X_train.drop(categ_colnames, axis=1)
        X_test.drop(categ_colnames, axis=1)                                          #yyyyyyy

        # normalize necessary columns
        normalizer = StandardScaler()
        X_train.iloc[:, 0:10] = normalizer.fit_transform(X_train.iloc[:, 0:10])
        X_test.iloc[:, 0:10] = normalizer.fit_transform(X_test.iloc[:, 0:10])         #yyyyyyy

        # merge encoded vs original columns
        X_train = pd.DataFrame(np.c_[encoded_cols, X_train])
        X_test = pd.DataFrame(np.c_[encoded_cols_1, X_test])                              #yyyyyyy
    except:
        pass

    # TODO: Generate polynomial features of order 2 from the current features.
    # If the input values are [a, b, c, d], you should generate
    # [a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2]. You can generate such polynomial
    # features either manually, or using
    # `sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)`.
    poly = PolynomialFeatures(2, include_bias=False)
    start_col = X_train.shape[1]
    X_train = pd.DataFrame(poly.fit_transform(X_train))
    X_train = X_train.iloc[:, start_col:]
    start_col_1 = X_test.shape[1]                                                         #yyyyyyy
    X_test = pd.DataFrame(poly.fit_transform(X_test))                                     #yyyyyyy
    X_test = X_test.iloc[:, start_col_1:]                                                  #yyyyyyy

    # TODO: You can wrap all the feature processing steps into one transformer
    # by using `sklearn.pipeline.Pipeline`. Although not strictly needed, it is
    # usually comfortable.

    # TODO: Fit the feature processing steps on the training data.
    # Then transform the training data into `train_data` (you can do both these
    # steps using `fit_transform`), and transform testing data to `test_data`.
    train_data = np.asarray(X_train)
    test_data = np.asarray(X_test)
    #print(train_data.shape)
    #print(test_data.shape)

    return train_data, test_data


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(args)
    '''for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 60))))'''

