#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import heapq as hq

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bootstrapping", default=False, action="store_true", help="Perform data bootstrapping")
parser.add_argument("--feature_subsampling", default=1, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Use the wine dataset
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    generator = np.random.RandomState(args.seed)

    class Node:
        def __init__(self, entropy, num_samples, num_samples_per_class, predicted_class):
            self.entropy = entropy
            self.num_samples = num_samples
            self.num_samples_per_class = num_samples_per_class
            self.predicted_class = predicted_class
            self.feature_index = 0
            self.threshold = 0
            self.left = None
            self.right = None

    class DecisionTreeClassifier:
        def __init__(self):
            self.max_depth = args.max_depth


        def _best_split(self, X, y):

            # Count of each class in the current node.
            num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

            # Entropy of current node.
            best_entropy = sum((n / len(y)) * (1 - (n / len(y))) for n in num_parent)

            if best_entropy == 0 or len(y) < args.min_to_split:
                return None, None

            best_idx, best_thr = None, None

            # Loop through all features.
            for idx in range(self.n_features_):
                # Sort data along selected feature.
                thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

                num_left = [0] * self.n_classes_
                num_right = num_parent.copy()

                # Possible split positions
                for i in range(1, len(y)):
                    c = classes[i - 1]
                    num_left[c] += 1
                    num_right[c] -= 1

                    entropy_left = -1 * sum(
                        (num_left[x] / i) * np.log(num_left[x] / i) for x in range(self.n_classes_) if
                        (num_left[x] / i) != 0)
                    entropy_right = -1 * sum(
                        (num_right[x] / (len(y) - i)) * np.log((num_right[x] / (len(y) - i))) for x in
                        range(self.n_classes_) if (num_right[x] / (len(y) - i)) != 0)

                    # Entropy of a split is the weighted average of children
                    entropy = (i * entropy_left + (len(y) - i) * entropy_right) / len(y)

                    if thresholds[i] == thresholds[i - 1]:
                        continue

                    if entropy < best_entropy:
                        best_entropy = entropy
                        best_idx = idx
                        best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

            return best_idx, best_thr

        def fit(self, X, y):
            self.n_classes_ = len(set(y))
            self.n_features_ = X.shape[1]
            self.tree_ = self._grow_tree(X, y)

        def _grow_tree(self, X, y, depth=0):
            num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
            predicted_class = np.argmax(num_samples_per_class)

            # Entropy of node.
            entropy = -1 * sum(
                    (n / len(y)) * np.log(n / len(y)) for n in num_samples_per_class if (n / len(y)) != 0)

            node = Node(
                entropy=entropy,
                num_samples=len(y),
                num_samples_per_class=num_samples_per_class,
                predicted_class=predicted_class,
            )

            if self.max_depth == None:
                self.max_depth = 1000

            if depth < self.max_depth:
                idx, thr = self._best_split(X, y)
                if idx is not None:
                    indices_left = X[:, idx] < thr
                    X_left, y_left = X[indices_left], y[indices_left]
                    X_right, y_right = X[~indices_left], y[~indices_left]
                    node.feature_index = idx
                    node.threshold = thr
                    node.left = self._grow_tree(X_left, y_left, depth + 1)
                    node.right = self._grow_tree(X_right, y_right, depth + 1)

            return node

        def predict(self, X):
            return [self._predict(inputs) for inputs in X]

        def _predict(self, inputs):
            node = self.tree_
            while node.left:
                if inputs[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            return node.predicted_class

    # Initiate obj
    my_Tree = DecisionTreeClassifier()
    my_Tree.fit(train_data, train_target)
    train_t, test_t = my_Tree.predict(train_data), my_Tree.predict(test_data)

    # TODO: Finally, measure the training and testing accuracy.
    train_accuracy = sklearn.metrics.accuracy_score(train_t, train_target)
    test_accuracy = sklearn.metrics.accuracy_score(test_t, test_target)

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    #print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    #print("Test accuracy: {:.1f}%".format(100 * test_accuracy))