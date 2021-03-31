#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np
import sklearn.compose
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.kernel_approximation
import sklearn.neural_network
import sklearn.svm
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.neighbors

class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)
        self.data = self.data.reshape([-1, 28*28]).astype(np.float)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--augment", default=False, action="store_true", help="Augment during training")
parser.add_argument("--models", default=1, type=int, help="Model to train")
parser.add_argument("--iterations", default=15, type=int, help="Training iterations")
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")

import scipy.ndimage
def augment(x):
    x = x.reshape(28, 28)
    x = scipy.ndimage.zoom(x.reshape(28, 28), (np.random.uniform(0.86, 1.2), np.random.uniform(0.86, 1.2)))
    x = np.pad(x, ((2, 2), (2, 2)))
    os = [np.random.randint(size - 28 + 1) for size in x.shape]
    x = x[os[0]:os[0] + 28, os[1]:os[1] + 28]
    x = scipy.ndimage.rotate(x, np.random.uniform(-15, 15), reshape=False)
    x = np.clip(x, 0, 1)
    return x.reshape(-1)

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        model = sklearn.pipeline.Pipeline([
            ("scaler", sklearn.preprocessing.MinMaxScaler()),
            ("MLPs", sklearn.ensemble.VotingClassifier([
                ("MLP{}".format(i), sklearn.neural_network.MLPClassifier(
                    tol=0, verbose=1, alpha=0, hidden_layer_sizes=(500), max_iter=1 if args.augment else args.iterations))
                for i in range(args.models)
            ], voting="soft")),
        ])
        model.fit(train.data, train.target)

        if args.augment:
            import multiprocessing
            pool = multiprocessing.Pool(16)
            for mlp in model["MLPs"].estimators_:
                for epoch in range(args.iterations - 1):
                    print("Augmenting data for epoch {}...".format(epoch), end="", flush=True)
                    augmented_data = pool.map(augment, model["scaler"].transform(train.data))
                    print("Done")
                    mlp.partial_fit(augmented_data, train.target)

        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained MLPClassifier is in `mlp` variable.
        # mlp._optimizer = None
        # for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        # for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)
        for mlp in model["MLPs"].estimators_:
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