#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Use the wine dataset
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    class Node:
        def __init__(self, gini_or_entropy, num_samples, num_samples_per_class, predicted_class):
            self.gini_or_entropy = gini_or_entropy
            self.num_samples = num_samples
            self.num_samples_per_class = num_samples_per_class
            self.predicted_class = predicted_class
            self.feature_index = 0
            self.threshold = 0
            self.left = None
            self.right = None


    class DecisionTreeClassifier:
        def __init__(self, max_depth=None):
            self.max_depth = max_depth

        def _best_split(self, args, X, y):

            # Need at least two elements to split a node.
            if len(y) < args.min_to_split:  # +++
                return None, None

            # Count of each class in the current node.
            num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

            # Gini/Entropy of current node.
            if args.criterion == "gini":
                best_gini = sum((n / len(y)) * (1 - (n / len(y))) for n in num_parent)
            else:
                best_gini = -1 * sum((n / len(y)) * np.log(n / len(y)) for n in num_parent if (n / len(y)) != 0)

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

                    if args.criterion == "gini":
                        gini_left = sum((num_left[x] / i) * (1 - (num_left[x] / i)) for x in range(4))
                        gini_right = sum(
                            (num_right[x] / (len(y) - i)) * (1 - (num_right[x] / (len(y) - i))) for x in range(4))
                    else:
                        gini_left = -1 * sum(
                            (num_left[x] / i) * np.log(num_left[x] / i) for x in range(4) if (num_left[x] / i) != 0)
                        gini_right = -1 * sum(
                            (num_right[x] / (len(y) - i)) * np.log((num_right[x] / (len(y) - i))) for x in range(4) if
                            (num_right[x] / (len(y) - i)) != 0)

                    # The Gini/Entropy of a split is the weighted average of children
                    gini = (i * gini_left + (len(y) - i) * gini_right) / len(y)

                    # The following condition is to make sure we don't try to split two
                    # points with identical values for that feature, as it is impossible
                    # (both have to end up on the same side of a split).
                    if thresholds[i] == thresholds[i - 1]:
                        continue

                    if gini < best_gini:
                        best_gini = gini
                        best_idx = idx
                        best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

            return best_idx, best_thr


        def fit(self, X, y):
            """Build decision tree classifier."""
            self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
            self.n_features_ = X.shape[1]
            self.tree_ = self._grow_tree(X, y)


        def _grow_tree(self, X, y, depth=0):
            """Build a decision tree by recursively finding the best split."""
            # Population for each class in current node. The predicted class is the one with
            # largest population.
            num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
            predicted_class = np.argmax(num_samples_per_class)
            # Gini/Entropy of node.
            if args.criterion == "gini":
                gini_or_entropy = sum((n / len(y)) * (1 - (n / len(y))) for n in num_samples_per_class)
            else:
                gini_or_entropy = -1 * sum((n / len(y)) * np.log(n / len(y)) for n in num_samples_per_class if (n / len(y)) != 0)


            node = Node(
                # TODO: add self._gini() function
                gini_or_entropy=gini_or_entropy,
                num_samples=len(y),
                num_samples_per_class=num_samples_per_class,
                predicted_class=predicted_class,
            )

            # Split recursively until maximum depth is reached.
            if depth < self.max_depth:
                idx, thr = self._best_split(args, X, y)
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
            """Predict class for a single sample."""
            node = self.tree_
            while node.left:
                if inputs[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            return node.predicted_class

    my_Tree = DecisionTreeClassifier()

    # TODO: Finally, measure the training and testing accuracy.
    train_accuracy = None
    test_accuracy = None

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    #print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    #print("Test accuracy: {:.1f}%".format(100 * test_accuracy))