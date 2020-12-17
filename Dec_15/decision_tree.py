#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import heapq as hq

# from sklearn._tree import BestFirstTreeBuilder
from sklearn.tree import DecisionTreeClassifier

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
        def __init__(self, X, y, gini_or_entropy, num_samples, num_samples_per_class, predicted_class, node_score,
                     node_idx, node_thr,
                     node_classes, birth_time):
            self.node_score = node_score
            self.birth_time = birth_time
            self.gini_or_entropy = gini_or_entropy
            self.num_samples = num_samples
            self.num_samples_per_class = num_samples_per_class
            self.predicted_class = predicted_class
            self.feature_index = 0
            self.threshold = 0
            self.left = None
            self.right = None
            self.X = X
            self.y = y
            self.node_idx = node_idx
            self.node_thr = node_thr
            self.node_classes = node_classes
            if self.node_score is None:
                self.node_score = -1000

        def __lt__(self, other):
            return self.birth_time < other.birth_time if self.node_score == other.node_score else self.node_score > other.node_score

    class DecisionTreeClassifier:
        def __init__(self):
            self.max_depth = args.max_depth
            self.max_leaves = args.max_leaves
            self.frontiers = []

        def _best_split(self, X, y):

            # Count of each class in the current node.
            num_classes = [np.sum(y == c) for c in range(self.n_classes_)]

            # Check out --min_to_split condition
            if len(y) < args.min_to_split:
                return None, None, None, num_classes

            # Gini/Entropy of current node.
            if args.criterion == "gini":
                best_score = sum((n / len(y)) * (1 - (n / len(y))) for n in num_classes)
            else:
                best_score = -1 * sum((n / len(y)) * np.log(n / len(y)) for n in num_classes if (n / len(y)) != 0)

            init_score = best_score

            best_idx, best_thr = None, None

            # Loop through all features.
            for idx in range(self.n_features_):
                # Sort data along selected feature.
                thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

                num_left = [0] * self.n_classes_
                num_right = num_classes.copy()

                # Possible split positions
                for i in range(1, len(y)):
                    c = classes[i - 1]
                    num_left[c] += 1
                    num_right[c] -= 1

                    if args.criterion == "gini":
                        score_left = sum((num_left[x] / i) * (1 - (num_left[x] / i)) for x in range(self.n_classes_))
                        score_right = sum((num_right[x] / (len(y) - i)) * (1 - (num_right[x] / (len(y) - i))) for x in
                                          range(self.n_classes_))
                    else:
                        score_left = -1 * sum(
                            (num_left[x] / i) * np.log(num_left[x] / i) for x in range(self.n_classes_) if
                            (num_left[x] / i) != 0)
                        score_right = -1 * sum(
                            (num_right[x] / (len(y) - i)) * np.log((num_right[x] / (len(y) - i))) for x in
                            range(self.n_classes_) if (num_right[x] / (len(y) - i)) != 0)

                    # The Gini/Entropy of a split is the weighted average of children
                    score = (i * score_left + (len(y) - i) * score_right) / len(y)

                    if thresholds[i] == thresholds[i - 1]:
                        continue

                    if score < best_score:
                        best_score = score
                        best_idx = idx
                        best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

            delta_gini = init_score - best_score
            return best_idx, best_thr, delta_gini, num_classes

        def fit(self, X, y):
            """Build decision tree classifier."""
            self.n_classes_ = len(set(y))
            self.n_features_ = X.shape[1]
            self.tree_ = self._grow_tree(X, y)

        def _single_node(self, new_X, new_y, birth_time):
            """Build a decision tree by recursively finding the best split."""
            # Population for each class in current node. The predicted class is the one with largest population.
            num_samples_per_class = [np.sum(new_y == i) for i in range(self.n_classes_)]
            predicted_class = np.argmax(num_samples_per_class)

            # Gini/Entropy of node.
            if args.criterion == "gini":
                gini_or_entropy = sum((n / len(new_y)) * (1 - (n / len(new_y))) for n in num_samples_per_class)
            else:
                gini_or_entropy = -1 * sum(
                    (n / len(new_y)) * np.log(n / len(new_y)) for n in num_samples_per_class if (n / len(new_y)) != 0)

            node_idx, node_thr, node_score, node_classes = self._best_split(new_X, new_y)

            single_node = Node(
                X=new_X,
                y=new_y,
                gini_or_entropy=gini_or_entropy,
                num_samples=len(new_y),
                num_samples_per_class=num_samples_per_class,
                predicted_class=predicted_class,
                node_score=node_score,
                node_idx=node_idx,
                node_thr=node_thr,
                node_classes=node_classes,
                birth_time=birth_time
            )
            return single_node

        def _grow_tree(self, X, y, depth=0, birth_time=0):
            root_node = self._single_node(X, y, birth_time=0)

            hq.heappush(self.frontiers, root_node)
            max_leaves = len(self.frontiers)

            if self.max_depth is None:
                self.max_depth = 1000

            if self.max_leaves is None:
                if depth < self.max_depth:
                    idx, thr, _, _ = self._best_split(X, y)
                    if idx is not None:
                        indices_left = X[:, idx] < thr
                        X_left, y_left = X[indices_left], y[indices_left]
                        X_right, y_right = X[~indices_left], y[~indices_left]
                        print(len(y_right), len(y_left))
                        root_node.feature_index = idx
                        root_node.threshold = thr
                        root_node.left = self._grow_tree(X_left, y_left, depth + 1)
                        root_node.right = self._grow_tree(X_right, y_right, depth + 1)
            else:

                while max_leaves < self.max_leaves:
                    if depth < self.max_depth:
                        mynode = hq.heappop(self.frontiers)
                        if mynode.node_score == -1000:
                            break
                        idx, thr, X, y = mynode.node_idx, mynode.node_thr, mynode.X, mynode.y
                        X = np.array(X)
                        y = np.array(y)
                        if idx is not None:
                            indices_left = X[:, idx] < thr
                            X_left, y_left = X[indices_left], y[indices_left]
                            X_right, y_right = X[~indices_left], y[~indices_left]
                            mynode.feature_index = idx
                            mynode.threshold = thr
                            mynode.left = self._single_node(np.array(X_left), np.array(y_left), birth_time + 1)
                            mynode.right = self._single_node(np.array(X_right), np.array(y_right), birth_time + 2)
                            print("Left vs Right score:", mynode.left.node_score, mynode.right.node_score)
                            print("Left vs Right classes:", mynode.left.node_classes, mynode.right.node_classes)
                            print("Left vs Right length:", len(mynode.left.y), len(mynode.right.y), "\n =================")
                            hq.heappush(self.frontiers, mynode.left)
                            hq.heappush(self.frontiers, mynode.right)
                            depth += 1
                            birth_time += 1
                        max_leaves = len(self.frontiers)
                    else:
                        break
            return root_node

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

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))
