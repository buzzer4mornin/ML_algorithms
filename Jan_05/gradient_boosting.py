#!/usr/bin/env python3
import argparse
# coding:utf-8
import numpy as np
# logistic function
from scipy.special import expit

from .base import BaseEstimator
from .tree import Tree
import numpy as np
import sklearn.datasets
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

def main(args):
    # Use the given dataset
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    classes = np.max(target) + 1

    def mse_criterion(y, splits):
        y_mean = np.mean(y)
        return -sum([np.sum((split - y_mean) ** 2) * (float(split.shape[0]) / y.shape[0]) for split in splits])

    """
    References:
    https://arxiv.org/pdf/1603.02754v3.pdf
    http://www.saedsayad.com/docs/xgboost.pdf
    https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf
    http://stats.stackexchange.com/questions/202858/loss-function-approximation-with-taylor-expansion
    """

    class Loss:
        """Base class for loss functions."""

        def __init__(self, regularization=1.0):
            self.regularization = regularization

        def grad(self, actual, predicted):
            """First order gradient."""
            raise NotImplementedError()

        def hess(self, actual, predicted):
            """Second order gradient."""
            raise NotImplementedError()

        def approximate(self, actual, predicted):
            """Approximate leaf value."""
            return self.grad(actual, predicted).sum() / (self.hess(actual, predicted).sum() + self.regularization)

        def transform(self, pred):
            """Transform predictions values."""
            return pred

        def gain(self, actual, predicted):
            """Calculate gain for split search."""
            nominator = self.grad(actual, predicted).sum() ** 2
            denominator = self.hess(actual, predicted).sum() + self.regularization
            return 0.5 * (nominator / denominator)

    class LogisticLoss(Loss):
        """Logistic loss."""

        def grad(self, actual, predicted):
            return actual * expit(-actual * predicted)

        def hess(self, actual, predicted):
            expits = expit(predicted)
            return expits * (1 - expits)

        def transform(self, output):
            # Apply logistic (sigmoid) function to the output
            return expit(output)

    class GradientBoosting(BaseEstimator):
        """Gradient boosting trees with Taylor's expansion approximation (as in xgboost)."""

        def __init__(self, n_estimators, learning_rate=0.1, max_features=10, max_depth=2, min_samples_split=10):
            self.min_samples_split = min_samples_split
            self.learning_rate = learning_rate
            self.max_depth = max_depth
            self.max_features = max_features
            self.n_estimators = n_estimators
            self.trees = []
            self.loss = None

        def fit(self, X, y=None):
            self._setup_input(X, y)
            self.y_mean = np.mean(y)
            self._train()

        def _train(self):
            # Initialize model with zeros
            y_pred = np.zeros(self.n_samples, np.float32)

            for n in range(self.n_estimators):
                residuals = self.loss.grad(self.y, y_pred)
                tree = Tree(regression=True, criterion=mse_criterion)
                # Pass multiple target values to the tree learner
                targets = {
                    # Residual values
                    "y": residuals,
                    # Actual target values
                    "actual": self.y,
                    # Predictions from previous step
                    "y_pred": y_pred,
                }
                tree.train(
                    self.X,
                    targets,
                    max_features=self.max_features,
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    loss=self.loss,
                )
                predictions = tree.predict(self.X)
                y_pred += self.learning_rate * predictions
                self.trees.append(tree)

        def _predict(self, X=None):
            y_pred = np.zeros(X.shape[0], np.float32)

            for i, tree in enumerate(self.trees):
                y_pred += self.learning_rate * tree.predict(X)
            return y_pred

        def predict(self, X=None):
            return self.loss.transform(self._predict(X))

    class GradientBoostingClassifier(GradientBoosting):
        def fit(self, X, y=None):
            # Convert labels from {0, 1} to {-1, 1}
            y = (y * 2) - 1
            self.loss = LogisticLoss()
            super(GradientBoostingClassifier, self).fit(X, y)



    # TODO: Finally, measure your training and testing accuracies when
    # using 1, 2, ..., `args.trees` of the created trees.
    #
    # To perform a prediction using t trees, compute the y_t(x_i) and return the
    # class with the highest value (and the smallest class if there is a tie).
    train_accuracies = []
    test_accuracies = []

    return train_accuracies, test_accuracies

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracies, test_accuracies = main(args)

    for i, (train_accuracy, test_accuracy) in enumerate(zip(train_accuracies, test_accuracies)):
        print("Using {} trees, train accuracy: {:.1f}%, test accuracy: {:.1f}%".format(
            i + 1, 100 * train_accuracy, 100 * test_accuracy))