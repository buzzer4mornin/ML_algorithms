#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request

import pandas as pd
import numpy as np
import sklearn.compose
import sklearn.dummy
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing

class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
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
parser.add_argument("--cv", default=5, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--model", default="lr", type=str, help="Model to use")
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        print(pd.DataFrame(train.data).info())
        exit()
        # TODO: Train a model on the given dataset and store it in `model`.
        if args.model in ["most_frequent", "stratified"]:
            model = sklearn.dummy.DummyClassifier(strategy=args.model)
        elif args.model == "gbt":
            model = sklearn.ensemble.GradientBoostingClassifier(max_depth=6, n_estimators=200).fit(train.data, train.target)
        else:
            if args.model == "lr":
                model = [
                    ("poly", sklearn.preprocessing.PolynomialFeatures(2)),
                    ("lr_cv", sklearn.linear_model.LogisticRegressionCV(Cs=np.geomspace(0.001, 1000, 7), max_iter=100)),
                ]
            elif args.model == "adalr":
                model = [
                    ("poly", sklearn.preprocessing.PolynomialFeatures(2)),
                    ("ada_lr_cv", sklearn.ensemble.AdaBoostClassifier(sklearn.linear_model.LogisticRegression(C=1), n_estimators=50)),
                ]
            elif args.model == "baglr":
                model = [
                    ("poly", sklearn.preprocessing.PolynomialFeatures(2)),
                    ("bag_lr_cv", sklearn.ensemble.BaggingClassifier(sklearn.linear_model.LogisticRegression(C=1), n_estimators=50)),
                ]
            elif args.model == "badlr":
              model = [("lr", sklearn.linear_model.LogisticRegression())]
            elif args.model == "mlp":
                model = [
                    ("MLP_ensemble", sklearn.ensemble.VotingClassifier([
                        ("MLP{}".format(i), sklearn.neural_network.MLPClassifier(tol=0, learning_rate_init=0.01, max_iter=200, hidden_layer_sizes=(300,200,100), activation="relu", solver="adam", verbose=1)) for i in range(3)])),
                ]
            int_columns = np.all(train.data.astype(int) == train.data, axis=0)
            model = sklearn.pipeline.Pipeline([
                ("preprocess", sklearn.compose.ColumnTransformer([
                    ("onehot", sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore"), int_columns),
                    ("scaler", sklearn.preprocessing.StandardScaler(), ~int_columns),
                ]))
            ] + model)

        if args.cv:
            scores = sklearn.model_selection.cross_val_score(model, train.data, train.target, cv=args.cv)
            print("Cross-validation with {} folds: {:.2f} +-{:.2f}".format(args.cv, 100 * scores.mean(), 100 * scores.std()))

        model.fit(train.data, train.target)

        if args.model == "mlp":
            # Remove moments and convert weights to 16bit floats.
            for mlp in model["MLP_ensemble"].estimators_:
                mlp._optimizer = None
                for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
                for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)