#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.metrics
import sklearn.neural_network

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--data_size", default=50, type=int, help="Data size")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=3, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--iterations", default=200, type=int, help="Number of training iterations")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization weight")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--nn_reference", default=False, action="store_true", help="Show NN solution too")

def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    train_data = np.linspace(-1, 1, args.data_size)
    train_target = np.sin(5 * train_data) + generator.normal(scale=0.25, size=args.data_size) + 1

    test_data = np.linspace(-1.2, 1.2, 2 * args.data_size)
    test_target = np.sin(5 * test_data) + 1

    betas = np.zeros(args.data_size)

    # TODO: Perform `args.iterations` of SGD-like updates, but in dual formulation
    # using `betas` as weights of individual training examples.
    #
    # We assume the primary formulation of our model is
    #   y = phi(x)^T w + bias
    # and the loss in the primary problem is batched MSE with L2 regularization:
    #   L = sum_{i \in B} 1/|B| * [1/2 * (phi(x_i)^T w + bias - target_i)^2] + 1/2 * args.l2 * w^2
    #
    # For `bias`, use explicitly the average of the training targets, and do
    # not update it futher during training.
    #
    # Instead of using feature map `phi` directly, we use a given kernel computing
    #   K(x, y) = phi(x)^T phi(y)
    # We consider the following `args.kernel`s:
    # - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    #
    # After each iteration, compute RMSE both on training and testing data.
    train_rmses, test_rmses = [], []

    bias = np.mean(train_target)

    def kernel(x, y):
        if args.kernel == "poly":
            return (args.kernel_gamma * x * y + 1) ** args.kernel_degree
        if args.kernel == "rbf":
            return np.exp(-args.kernel_gamma * ((x - y) ** 2))

    K = np.array([[kernel(x, y) for x in train_data] for y in train_data])
    Ktest = np.array([[kernel(x, y) for x in train_data] for y in test_data])
    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`, performing
        # batched updates to the `betas`. You can assume that `args.batch_size`
        # exactly divides `train_data.shape[0]`.
        for index in range(0, len(permutation), args.batch_size):
            batch = permutation[index:index + args.batch_size]
            assert len(batch) == args.batch_size

            betas[batch] -= args.learning_rate / args.batch_size * (K[batch] @ betas + bias - train_target[batch])
            betas -= args.learning_rate * args.l2 * betas

        # TODO: Append RMSE on training and testing data to `train_rmses` and
        # `test_rmses` after the iteration.
        train_rmses.append(sklearn.metrics.mean_squared_error(train_target, K @ betas + bias, squared=False))
        test_rmses.append(sklearn.metrics.mean_squared_error(test_target, Ktest @ betas + bias, squared=False))

        if (iteration + 1) % 10 == 0:
            print("Iteration {}, train RMSE {:.2f}, test RMSE {:.2f}".format(
                iteration + 1, train_rmses[-1], test_rmses[-1]))

    if args.plot:
        import matplotlib.pyplot as plt
        # If you want the plotting to work (not required for ReCodEx), compute the `test_predictions`.
        test_predictions = Ktest @ betas + bias

        plt.plot(train_data, train_target, "bo", label="Train target")
        plt.plot(test_data, test_target, "ro", label="Test target")
        plt.plot(test_data, test_predictions, "g-", label="Predictions")
        plt.legend()
        if args.nn_reference:
            for params in [{"activation": "tanh"}, {"activation": "relu"}]:
                plt.figure()
                mlp = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(500,), solver="lbfgs", alpha=0, max_iter=2000, **params).fit(np.expand_dims(train_data, -1), train_target)
                plt.plot(train_data, train_target, "bo", label="Train target")
                plt.plot(test_data, test_target, "ro", label="Test target")
                plt.plot(test_data, mlp.predict(np.expand_dims(test_data, -1)), "g-", label="Predictions")
                plt.legend()
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return train_rmses, test_rmses

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)