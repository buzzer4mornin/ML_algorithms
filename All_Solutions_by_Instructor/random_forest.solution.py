#!/usr/bin/env python3
import argparse
import subprocess

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

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
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")

class DecisionTree:
    class Node:
        def __init__(self, instances, prediction):
            self.is_leaf = True
            self.instances = instances
            self.prediction = prediction

        def split(self, feature, value, left, right):
            self.is_leaf = False
            self.feature = feature
            self.value = value
            self.left = left
            self.right = right

    def __init__(self, random_generator, feature_subsampling, max_depth):
        self._random_generator = random_generator
        self._feature_subsampling = feature_subsampling
        self._max_depth = max_depth

    def fit(self, data, targets):
        self._data = data
        self._targets = targets

        self._root = self._create_leaf(np.arange(len(self._data)))
        self._split_recursively(self._root, 0)

        return self

    def predict(self, data):
        results = np.zeros(len(data), dtype=np.int32)
        for i in range(len(data)):
            node = self._root
            while not node.is_leaf:
                node = node.left if data[i][node.feature] <= node.value else node.right
            results[i] = node.prediction

        return results

    def _split_recursively(self, node, depth):
        if not self._can_split(node, depth):
            return

        feature, value, left, right = self._best_split(node)
        node.split(feature, value, self._create_leaf(left), self._create_leaf(right))
        self._split_recursively(node.left, depth + 1)
        self._split_recursively(node.right, depth + 1)

    def _can_split(self, node, depth):
        return (
            (self._max_depth is None or depth < self._max_depth) and
            not np.array_equiv(self._targets[node.instances], node.prediction)
        )

    def _best_split(self, node):
        best_criterion = None
        for feature in np.where(self._random_generator.uniform(size=self._data.shape[1]) <= self._feature_subsampling)[0]:
            node_features = self._data[node.instances, feature]
            separators = np.unique(node_features)
            for i in range(len(separators) - 1):
                value = (separators[i] + separators[i + 1]) / 2
                left, right = node.instances[node_features <= value], node.instances[node_features > value]
                criterion = self._criterion(left) + self._criterion(right)
                if best_criterion is None or criterion < best_criterion:
                    best_criterion, best_feature, best_value, best_left, best_right = \
                        criterion, feature, value, left, right
        print(best_feature, best_value, best_criterion)

        return best_feature, best_value, best_left, best_right

    def _criterion(self, instances):
        # We use the entropy
        bins = np.bincount(self._targets[instances])
        bins = bins[np.nonzero(bins)]
        return -np.sum(bins * np.log(bins / len(instances)))

    def _create_leaf(self, instances):
        # Create a new leaf, together with its prediction (the most frequent class)
        return self.Node(instances, np.argmax(np.bincount(self._targets[instances])))


class RandomForest:
    def __init__(self, random_generator, trees, bootstrapping, feature_subsampling, max_depth):
        self._random_generator = random_generator
        self._num_trees = trees
        self._bootstrapping = bootstrapping
        self._feature_subsampling = feature_subsampling
        self._max_depth = max_depth

    def fit(self, data, targets):
        self._classes = np.max(targets) + 1
        self._trees = []
        for i in range(self._num_trees):
            data_indices = self._random_generator.choice(len(data), size=len(data)) if self._bootstrapping else np.arange(len(data))
            self._trees.append(DecisionTree(
                self._random_generator, self._feature_subsampling, self._max_depth
            ).fit(
                data[data_indices], targets[data_indices]
            ))

    def predict(self, data):
        results = np.zeros((len(data), self._classes), np.int32)
        for tree in self._trees:
            for index, prediction in enumerate(tree.predict(data)):
                results[index, prediction] += 1
        return np.argmax(results, axis=1)

def main(args):
    # Use the wine dataset
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a random forest on the trainining data.
    #
    # For determinism, create a generator
    #   generator = np.random.RandomState(args.seed)
    # at the beginning and then use this instance for all random number generation.
    #
    # Use a simplified decision tree from the `decision_tree` assignment:
    # - use `entropy` as the criterion
    # - use `max_depth` constraint, so split a node only if:
    #   - its depth is less than `args.max_depth`
    #   - the criterion is not 0 (the corresponding instance targetsare not the same)
    # When splitting nodes, proceed in the depth-first order, splitting all nodes
    # in left subtrees before nodes in right subtrees.
    #
    # Additionally, implement:
    # - feature subsampling: when searching for the best split, try only
    #   a subset of features. When splitting a node, start by generating
    #   a feature mask using
    #     generator.uniform(size=number_of_features) <= feature_subsampling
    #   which gives a boolean value for every feature, with `True` meaning the
    #   feature is used during best split search, and `False` it is not.
    #   (When feature_subsampling == 1, all features are used, but the mask
    #   should still be generated.)
    #
    # - train a random forest consisting of `args.trees` decision trees
    #
    # - if `args.bootstrapping` is set, right before training a decision tree,
    #   create a bootstrap sample of the training data using the following indices
    #     indices = generator.choice(len(train_data), size=len(train_data))
    #   and if `args.bootstrapping` is not set, use the original training data.
    #
    # During prediction, use voting to find the most frequent class for a given
    # input, choosing the one with smallest class index in case of a tie.
    random_forest = RandomForest(
        np.random.RandomState(args.seed), args.trees, args.bootstrapping, args.feature_subsampling, args.max_depth)
    random_forest.fit(train_data, train_target)

    # TODO: Finally, measure the training and testing accuracy.
    train_accuracy = sklearn.metrics.accuracy_score(train_target, random_forest.predict(train_data))
    test_accuracy = sklearn.metrics.accuracy_score(test_target, random_forest.predict(test_data))

    if args.plot:
        # Plot the final tree using graphviz
        feature_names = sklearn.datasets.load_wine().feature_names
        dot = ["digraph Tree {node [shape=box]; bgcolor=invis;"]
        def plot(index, tree, node, parent):
            if parent is not None: dot.append("{} -> {}".format(parent, index))
            dot.append("{} [fontname=\"serif\"; label=\"{}c_entropy = {:.2f}\\ninstances = {}\\ncounts = [{}]\"];".format(
                index, "{} <= {:.3f}\\n".format(feature_names[node.feature], node.value) if not node.is_leaf else "",
                tree._criterion(node.instances), len(node.instances),
                ",".join(map(str, np.bincount(tree._targets[node.instances], minlength=3)))))
            if not node.is_leaf:
                index = plot(plot(index + 1, tree, node.left, index), tree, node.right, index)
            return index + 1
        index = 0
        for tree in random_forest._trees:
            index = plot(index, tree, tree._root, None)
        dot.append("}")
        subprocess.run(["dot", "-Txlib"] if args.plot is True else ["dot", "-Tsvg", "-o{}".format(args.plot)],
                       input="\n".join(dot), encoding="utf-8")

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))