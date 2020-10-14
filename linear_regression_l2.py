#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    X = np.asarray(dataset.data)
    Y = np.asarray(dataset.target)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_size,
                                                        random_state=args.seed)

    lambdas = np.geomspace(0.01, 100, num=500)
    # TODO: Using `sklearn.linear_model.Ridge`, fit the train set using
    # L2 regularization, employing above defined lambdas.
    # For every model, compute the root mean squared error
    # (do not forget `sklearn.metrics.mean_squared_error`) and return the
    # lambda producing lowest test error.
    best_rmse = 100
    rmses = []
    for l in lambdas:
        clf = Ridge(alpha=round(l, 2), tol=0.001, solver='auto')
        clf.fit(X_train, Y_train)
        predicted_Y = clf.predict(X_test)
        training_error = clf.predict(X_train)
        rmse = np.math.sqrt(sklearn.metrics.mean_squared_error(predicted_Y, Y_test))
        rmses.append(rmse)
        if rmse < best_rmse:
            best_lambda = round(l, 2)
            best_rmse = rmse


    if args.plot:
        # This block is not required to pass in ReCodEx, however, it is useful
        # to learn to visualize the results.

        # If you collect the respective results for `lambdas` to an array called `rmse`,
        # the following lines will plot the result if you add `--plot` argument.
        import matplotlib.pyplot as plt
        plt.plot(lambdas, rmses)
        plt.xscale("log")
        plt.xlabel("L2 regularization strength")
        plt.ylabel("RMSE")
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return best_lambda, best_rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    best_lambda, best_rmse = main(args)
    print("{:.2f} {:.2f}".format(best_lambda, best_rmse))