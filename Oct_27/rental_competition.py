#!/usr/bin/env python3

# Team:
# 2f67b427-a885-11e7-a937-00505601122b
# b030d249-e9cb-11e9-9ce9-00505601122b
# 3351ff04-3f62-11e9-b0fd-00505601122b

import argparse
import lzma
import os
import pickle
import urllib.request
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer


class Dataset:

    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        X = pd.DataFrame(train.data)
        y = pd.DataFrame(train.target)

        # Set column names
        col_names = ['season', 'year', 'month', 'hour', 'holiday', 'day_week', 'work_day',
                     'weather', 'temp', 'feel_temp', 'humidity', 'windspeed']
        X.columns = col_names

        # Drop unuseful columns [conclusion from Exploratory Data Analysis]
        X.drop(['season', 'feel_temp', 'day_week'], axis=1, inplace=True)

        # OneHotEncode some columns [conclusion from Exploratory Data Analysis]
        one_hots = ["month", "hour", "weather"]

        # Preparing for Pipeline
        col_trans = ColumnTransformer([('1hot', OneHotEncoder(sparse=False), one_hots)],
                                      remainder='passthrough')
        poly = PolynomialFeatures(3, include_bias=False)

        # Set Pipeline and Fit&Transform train data
        pipeline = sklearn.pipeline.Pipeline([('col_trans', col_trans), ('poly', poly)])
        X = pipeline.fit_transform(X)

        # TODO: Train a model on the given dataset and store it in `model`.
        """Fit Lasso Regression on Training Data"""
        # best rmse = 56.67, best_alfa = 0.2 --> found out from K-fold CrossVal
        model = Lasso(alpha=0.2, tol=0.001)
        model.fit(X, y)

        # Serialize OneHotEncoder.
        with lzma.open("pipeline.model", "wb") as model_file:
            pickle.dump(pipeline, model_file)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)
        X = pd.DataFrame(test.data)

        # Set column names
        col_names = ['season', 'year', 'month', 'hour', 'holiday', 'day_week', 'work_day',
                     'weather', 'temp', 'feel_temp', 'humidity', 'windspeed']
        X.columns = col_names

        # Drop unuseful columns
        X.drop(['season', 'feel_temp', 'day_week'], axis=1, inplace=True)

        with lzma.open("pipeline.model", "rb") as model_file:
            pipeline = pickle.load(model_file)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Transform test data
        X = pipeline.transform(X)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = np.array(model.predict(X))

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
