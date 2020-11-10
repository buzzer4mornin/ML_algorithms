#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
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

    '''def Lp_norm(A, B, p):  # A vector B matrix
        if p == 1:
            norm = np.sum(np.absolute(A - B), axis=1)
        elif p == 2:
            norm = (A - B) * (A - B)
            norm = np.sum(norm, axis=1)
            norm = np.sqrt(norm)
        elif p == 3:
            norm = np.absolute(A - B) * (A - B) * (A - B)
            norm = np.cbrt(np.sum(norm, axis=1))
        return norm  # vraci vektor vzdalenosti vektoru A od matice B'''

    def distance(a, b, p):
        d = np.linalg.norm((b - a), ord=p, axis=1)
        return d

    def minkowski_distance(a, b, p):
        distance = 0
        for d in range(len(a) - 1):
            distance += abs(a[d] - b[d]) ** p
        distance = distance ** (1 / p)
        return distance

    def get_neighbors(train_data, train_target, test_row, k, p, weight_mode):
        '''distances = list()
        for x, y in zip(train_data, train_target):
            dist = minkowski_distance(test_row, x, p)
            distances.append((y, dist))
        distances.sort(key=lambda tup: tup[1])'''

        '''d = list()
        for x in train_data:
            d.append(minkowski_distance(test_row, x, p))'''

        d = list(distance(test_row, train_data, p))


        neighbors = list()
        neighbors_w = list()
        for i in range(k):
            index = np.argmin(d)
            a = list(train_target).pop(index)
            b = d.pop(index)
            neighbors.append(a)
            neighbors_w.append(b)
            #neighbors.append(distances[i][0])
            #neighbors_w.append(distances[i][1])
        if weight_mode == "uniform":
            neighbors_w = [1 for _ in range(len(neighbors_w))]
        elif weight_mode == "inverse":
            neighbors_w = [1 / k for k in neighbors_w]
        elif weight_mode == "softmax":
            neighbors_w = stablesoftmax(neighbors_w)
        return np.array(neighbors), np.array(neighbors_w)

    def predict_classification(train_data, train_target, test_row, num_neighbors, p, weight_mode):
        neighbors, neighbors_w = get_neighbors(train_data, train_target, test_row, num_neighbors, p, weight_mode)
        result = weighted_mode(neighbors, neighbors_w)
        prediction = int(result[0])
        print(prediction)
        return prediction

    # TODO: Generate `test_predictions` with classes predicted for `test_data`.
    #
    # Find `args.k` nearest neighbors, choosing the ones with smallest train_data
    # indices in case of ties. Use the most frequent class (optionally weighted
    # by a given scheme described below) as prediction, again using the one with
    # smaller index when there are multiple classes with the same frequency.
    #
    # Use L_p norm for a given p (1, 2, 3) to measure distances.
    #
    # The weighting can be:
    # - "uniform": all nearest neighbors have the same weight
    # - "inverse": `1/distances` is used as weights
    # - "softmax": `softmax(-distances)` is uses as weights
    #
    # If you want to plot misclassified examples, you need to also fill `test_neighbors`
    # with indices of nearest neighbors; but it is not needed for passing in ReCodEx.
    test_predictions = []
    for test in test_data:
        prediction = predict_classification(train_data, train_target, test, args.k, args.p, args.weights)
        test_predictions.append(prediction)

    accuracy = sklearn.metrics.accuracy_score(test_target, test_predictions)

    '''if args.plot:
        import matplotlib.pyplot as plt
        examples = [[] for _ in range(10)]
        for i in range(len(test_predictions)):
            if test_predictions[i] != test_target[i] and not examples[test_target[i]]:
                examples[test_target[i]] = [test_data[i], *train_data[test_neighbors[i]]]
        examples = [[img.reshape(28, 28) for img in example] for example in examples if example]
        examples = [[example[0]] + [np.zeros_like(example[0])] + example[1:] for example in examples]
        plt.imshow(np.concatenate([np.concatenate(example, axis=1) for example in examples], axis=0), cmap="gray")
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")'''

    return accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("K-nn accuracy for {} nearest neighbors, L_{} metric, {} weights: {:.2f}%".format(
        args.k, args.p, args.weights, 100 * accuracy))