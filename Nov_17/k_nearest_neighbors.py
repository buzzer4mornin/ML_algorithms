#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import scipy
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
from sklearn.utils.extmath import weighted_mode


class MNIST:
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
parser.add_argument("--k", default=1, type=int, help="K nearest neighbors to consider")
parser.add_argument("--p", default=2, type=int, help="Use L_p as distance metric")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=500, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--train_size", default=1000, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--weights", default="uniform", type=str, help="Weighting to use (uniform/inverse/softmax)")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load MNIST data, scale it to [0, 1] and split it to train and test
    mnist = MNIST()
    mnist.data = sklearn.preprocessing.MinMaxScaler().fit_transform(mnist.data)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        mnist.data, mnist.target, stratify=mnist.target, train_size=args.train_size, test_size=args.test_size, random_state=args.seed)

    def stablesoftmax(x):
        """Compute the softmax of vector x in a numerically stable way."""
        expZ = np.exp(x - np.max(x))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def get_neighbors(train_data, train_target, test_row, k, p, weight_mode):
        #d = list()
        copy_t = list(train_target)
        #for x in train_data:
            #d.append(scipy.spatial.distance.minkowski(test_row, x, p))
        d = list(np.linalg.norm(test_row - train_data, ord=p, axis=-1))
        neighbors = list()
        neighbors_w = list()
        for i in range(k):
            index = np.argmin(d)
            a = copy_t.pop(index)
            b = d.pop(index)
            neighbors.append(a)
            neighbors_w.append(b)
        if weight_mode == "uniform":
            neighbors_w = [1 for _ in range(len(neighbors_w))]
        elif weight_mode == "inverse":
            neighbors_w = [1 / k for k in neighbors_w]
        elif weight_mode == "softmax":
            #neighbors_w = (-1) * neighbors_w
            #for i in range(list(neighbors_w)):
            #    neighbors_w[i] = (-1) * int(neighbors_w[i])
            neighbors_w = [element * (-1) for element in neighbors_w]
            print(neighbors_w)
            neighbors_w = stablesoftmax(neighbors_w)
        return np.array(neighbors), np.array(neighbors_w)

    def predict_classification(train_data, train_target, test_row, num_neighbors, p, weight_mode):
        neighbors, neighbors_w = get_neighbors(train_data, train_target, test_row, num_neighbors, p, weight_mode)

        result = weighted_mode(neighbors, neighbors_w)
        prediction = int(result[0])
        return prediction

    # TODO: Generate `test_predictions` with classes predicted for `test_data`.
    test_predictions = []
    for test in test_data:
        prediction = predict_classification(train_data, train_target, test, args.k, args.p, args.weights)
        test_predictions.append(prediction)
    accuracy = sklearn.metrics.accuracy_score(test_target, test_predictions)

    return accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("K-nn accuracy for {} nearest neighbors, L_{} metric, {} weights: {:.2f}%".format(
        args.k, args.p, args.weights, 100 * accuracy))