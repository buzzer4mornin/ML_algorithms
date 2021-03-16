#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import zipfile

import numpy as np
import pandas as pd

import sklearn.ensemble
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing


class Dataset:
    CLASSES = ["sitting", "sittingdown", "standing", "standingup", "walking"]

    def __init__(self,
                 name="human_activity_recognition.train.csv.xz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and if it contains column "class", split it to `targets`.
        self.data = pd.read_csv(name)
        if "class" in self.data:
            self.target = np.array([Dataset.CLASSES.index(target) for target in self.data["class"]], np.int32)
            self.data = self.data.drop("class", axis=1)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--cv", default=5, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--max_depth", default=5, type=int, help="Maximum tree depth")
parser.add_argument("--model", default="gbt", type=str, help="Model type")
parser.add_argument("--model_path", default="human_activity_recognition.model", type=str, help="Model path")
parser.add_argument("--subsample", default=0.5, type=float, help="Subsample data")
parser.add_argument("--trees", default=600, type=int, help="Number of trees to use")

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        model = sklearn.pipeline.Pipeline([
            ("scaler", sklearn.preprocessing.StandardScaler()),
            ("estimator", {
                "svm": sklearn.svm.SVC(verbose=1),
                "mlp": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(500), batch_size=50, verbose=1, early_stopping=True, validation_fraction=0.1),
                "gbt": sklearn.ensemble.GradientBoostingClassifier(n_estimators=args.trees, max_depth=args.max_depth, subsample=args.subsample, verbose=1),
                "rf": sklearn.ensemble.RandomForestClassifier(n_estimators=args.trees, verbose=1),
            }[args.model]),
        ])

        if args.cv:
            scores = sklearn.model_selection.cross_val_score(model, train.data, train.target, cv=args.cv, n_jobs=3)
            print("Cross-validation with {} folds: {:.2f} +-{:.2f}".format(args.cv, 100 * scores.mean(), 100 * scores.std()))

        model.fit(train.data, train.target)

        for mlp in [model["estimator"]] if args.model == "mlp" else []:
            mlp._optimizer = None
            for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
            for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list of a NumPy array.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)