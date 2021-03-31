#!/usr/bin/env python3
import argparse
import subprocess

import numpy as np
import sklearn.datasets
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--l2", default=1., type=float, help="L2 regularization factor")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=57, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--with_reference", default=False, action="store_true", help="Use reference implementation")

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

    def __init__(self, l2, max_depth):
        self._l2 = l2
        self._max_depth = max_depth

    def fit(self, data, gs, hs):
        self._data = data
        self._gs = gs
        self._hs = hs

        self._root = self._create_leaf(np.arange(len(self._data)))
        self._split_recursively(self._root, 0)

        return self

    def predict(self, data):
        results = np.zeros(len(data), dtype=np.float32)
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
            len(node.instances) > 1
        )

    def _best_split(self, node):
        best_criterion = None
        for feature in range(self._data.shape[1]):
            node_features = self._data[node.instances, feature]
            separators = np.unique(node_features)
            for i in range(len(separators) - 1):
                value = (separators[i] + separators[i + 1]) / 2
                left, right = node.instances[node_features <= value], node.instances[node_features > value]
                criterion = self._criterion(left) + self._criterion(right)
                if best_criterion is None or criterion < best_criterion:
                    best_criterion, best_feature, best_value, best_left, best_right = \
                        criterion, feature, value, left, right

        return best_feature, best_value, best_left, best_right

    def _criterion(self, instances):
        # Gradient boosting index
        return -0.5 * np.sum(self._gs[instances]) ** 2 / (np.sum(self._hs[instances]) + self._l2)

    def _create_leaf(self, instances):
        # Create a new leaf, together with its prediction (the most frequent class)
        return self.Node(instances, -np.sum(self._gs[instances]) / (np.sum(self._hs[instances]) + self._l2))

class GradientBoostedTrees:
    def __init__(self, trees, learning_rate, l2, max_depth):
        self._num_trees = trees
        self._learning_rate = learning_rate
        self._l2 = l2
        self._max_depth = max_depth

    def fit(self, data, targets):
        def softmax(z):
            softmax = np.exp(z - np.max(z, axis=-1, keepdims=True))
            return softmax / np.sum(softmax, axis=-1, keepdims=True)

        self._classes = np.max(targets) + 1
        self._trees = []
        predictions = np.zeros((len(data), self._classes), np.float32)
        for i in range(self._num_trees):
            probabilities = softmax(predictions)

            self._trees.append([])
            for c in range(self._classes):
                gs = probabilities[:, c] - (targets[:] == c)
                hs = probabilities[:, c] * (1 - probabilities[:, c])
                self._trees[-1].append(DecisionTree(self._l2, self._max_depth).fit(data, gs, hs))
                predictions[:, c] += self._learning_rate * self._trees[-1][c].predict(data)

    def predict(self, data, trees):
        predictions = np.zeros((len(data), self._classes), np.float32)
        for trees in self._trees[:trees]:
            for c, tree in enumerate(trees):
                predictions[:, c] += tree.predict(data)
        return np.argmax(predictions, axis=1)


def main(args):
    # Use the given dataset
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    classes = np.max(target) + 1

    # TODO: Create a gradient boosted trees on the classification training data.
    #
    # Notably, train for `args.trees` iteration. During iteration `t`:
    # - the goal is to train `classes` regression trees, each predicting
    #   raw weight for the corresponding class.
    # - compute the current predictions `y_t(x_i)` for every training example `i` as
    #     y_t(x_i)_c = \sum_{i=1}^t args.learning_rate * tree_{iter=i,class=c}.predict(x_i)
    #     (note that y_0 is zero)
    # - loss in iteration `t` is
    #     L = (\sum_i NLL(onehot_target_i, softmax(y_{t-1}(x_i) + trees_to_train_in_iter_t.predict(x_i)))) +
    #         1/2 * args.l2 * (sum of all node values in trees_to_train_in_iter_t)
    # - for every class `c`:
    #   - start by computing `g_i` and `h_i` for every training example `i`;
    #     the `g_i` is the first derivative of NLL(onehot_target_i_c, softmax(y_{t-1}(x_i))_c)
    #     with respect to y_{t-1}(x_i)_c, and the `h_i` is the second derivative of the same.
    #   - then, create a decision tree minimizing the above loss L. According to the slides,
    #     the optimum prediction for a given node T with training examples I_T is
    #       w_T = - (\sum_{i \in I_T} g_i) / (args.l2 + sum_{i \in I_T} h_i)
    #     and the value of the loss with the above prediction is
    #       c_GB = - 1/2 (\sum_{i \in I_T} g_i)^2 / (args.l2 + sum_{i \in I_T} h_i)
    #     which you should use as a splitting criterion.
    #
    # During tree construction, we split a node if:
    # - its depth is less than `args.max_depth`
    # - there is more than 1 example corresponding to it (this was covered by
    #     a non-zero criterion value in the previous assignments)
    gbt = GradientBoostedTrees(args.trees, args.learning_rate, args.l2, args.max_depth)
    if args.with_reference:
        gbt = sklearn.ensemble.GradientBoostingClassifier(
            n_estimators=args.trees, learning_rate=args.learning_rate, max_depth=args.max_depth, verbose=1, random_state=args.seed)
    gbt.fit(train_data, train_target)

    # TODO: Finally, measure your training and testing accuracies when
    # using 1, 2, ..., `args.trees` of the created trees.
    #
    # To perform a prediction using t trees, compute the y_t(x_i) and return the
    # class with the highest value (and the smallest class if there is a tie).
    train_accuracies, test_accuracies = [], []
    for trees in range(args.trees):
        train_accuracies.append(sklearn.metrics.accuracy_score(train_target, gbt.predict(train_data, trees + 1)))
        test_accuracies.append(sklearn.metrics.accuracy_score(test_target, gbt.predict(test_data, trees + 1)))

    if args.plot:
        # Plot the final tree using graphviz
        feature_names = getattr(sklearn.datasets, "load_{}".format(args.dataset))().feature_names
        dot = ["digraph Tree {node [shape=box]; bgcolor=invis;"]
        def plot(index, tree, node, parent, leaves):
            if parent is not None: dot.append("{} -> {}".format(parent, index))
            dot.append("{} [fontname=\"serif\"; label=\"{}c_gb = {:.2f}\\ninstances = {}\\nprediction={:.2f}\"];".format(
                index, "{} <= {:.3f}\\n".format(feature_names[node.feature], node.value) if not node.is_leaf else "",
                tree._criterion(node.instances), len(node.instances), node.prediction))
            if node.is_leaf:
                leaves.append(index)
            else:
                index = plot(plot(index + 1, tree, node.left, index, leaves), tree, node.right, index, leaves)
            return index + 1
        index = 0
        leaves = [[] for _ in range(classes)]
        for i, trees in enumerate(gbt._trees):
            for c, tree in enumerate(trees):
                dot.append("subgraph cluster_i{}_c{} {{ label=\"Tree {} for class {}\"".format(i, c, i + 1, c + 1))
                leaves[c] = leaves[c][:3]
                while leaves[c]: dot.append("{} -> {} [style=invis];".format(leaves[c].pop(), index))
                index = plot(index, tree, tree._root, None, leaves[c])
                dot.append("}")
        dot.append("}")
        subprocess.run(["dot", "-Txlib"] if args.plot is True else ["dot", "-Tsvg", "-o{}".format(args.plot)],
                       input="\n".join(dot), encoding="utf-8")

    return train_accuracies, test_accuracies

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracies, test_accuracies = main(args)

    for i, (train_accuracy, test_accuracy) in enumerate(zip(train_accuracies, test_accuracies)):
        print("Using {} trees, train accuracy: {:.1f}%, test accuracy: {:.1f}%".format(
            i + 1, 100 * train_accuracy, 100 * test_accuracy))