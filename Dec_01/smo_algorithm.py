#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")


# If you add more arguments, ReCodEx will keep them with your default values.

def kernel(args, x, y):
    # TODO: As in `kernel_linear_regression`, We consider the following `args.kernel`s:
    # - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    return [(args.kernel_gamma * np.dot(x, y) + 1) ** args.kernel_degree if args.kernel == "poly"
            else np.exp(-1 * args.kernel_gamma * ((x - y) @ (x - y)))]


# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.
def smo(args, train_data, train_target, test_data, test_target):
    # Create initial weights

    def predict_train(row):
        sums = 0
        for i in range(len(train_data)):
            sums += a[i] * train_target[i] * train_kernels[row, i]
        return sums + b

    def predict_test(row):
        sums = 0
        for i in range(len(train_data)):
            sums += a[i] * train_target[i] * test_kernels[row, i]
        return sums + b

    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)

    train_kernels = np.empty((len(train_data), len(train_data)), dtype=float)
    test_kernels = np.empty((len(test_data), len(train_data)), dtype=float)

    for x1, x1_ in enumerate(train_data):
        for x2, x2_ in enumerate(train_data):
            train_kernels[x1][x2] = kernel(args, x1_, x2_)[0]

    for x1, x1_ in enumerate(test_data):
        for x2, x2_ in enumerate(train_data):
            test_kernels[x1][x2] = kernel(args, x1_, x2_)[0]

    passes_without_as_changing = 0
    train_accs, test_accs = [], []

    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            j = j + (j >= i)

            E_i = predict_train(i) - train_target[i]
            E_j = predict_train(j) - train_target[j]

            if ((a[i] < (args.C - args.tolerance)) and (train_target[i] * E_i < -1 * args.tolerance)) or (a[i] > args.tolerance and (train_target[i] * E_i > args.tolerance)):
                d = 2 * train_kernels[i, j] - train_kernels[i, i] - train_kernels[j, j]
                # check if derivative > tol
                if d > -1 * args.tolerance:
                    continue
                else:
                    aj_ = a[j] - (train_target[j] * (E_i - E_j)) / d

                    # TODO: define range for [L,H] (both ti,tj equal or not)
                    if train_target[i] == train_target[j]:
                        L = max(0, a[i] + a[j] - args.C)
                        H = min(args.C, a[j] + a[i])
                    else:
                        L = max(0, a[j] - a[i])
                        H = min(args.C, args.C + a[j] - a[i])
                    # clip new aj
                    aj_ = np.clip(aj_, L, H)
                    if abs(aj_ - a[j]) < args.tolerance:
                        continue
                    else:
                        # ai, bias updates regarding KKT
                        ai_ = a[i] - train_target[i] * train_target[j] * (aj_ - a[j])
                        bi_ = b - E_i - train_target[i] * (ai_ - a[i]) * train_kernels[i, i] - train_target[j] * (aj_ - a[j]) * train_kernels[j, i]
                        bj_ = b - E_j - train_target[i] * (ai_ - a[i]) * train_kernels[i, j] - train_target[j] * (aj_ - a[j]) * train_kernels[j, j]

                        # new bias update
                        if args.tolerance < ai_ < (args.C - args.tolerance):
                            new_b = bi_
                        elif args.tolerance < aj_ < (args.C - args.tolerance):
                            new_b = bj_
                        else:
                            new_b = (bi_ + bj_) / 2
                        # updating ai,aj,b
                        a[i] = ai_
                        a[j] = aj_
                        b = new_b
                        as_changed += 1

        # TODO: After each iteration, measure the accuracy for both the
        # train set and the test set and append it to `train_accs` and `test_accs`.
        o = []
        for i in range(100):
            my_predict = predict_train(i)
            if my_predict >= 0:
                o.append(1)
            else:
                o.append(-1)
        my = sklearn.metrics.accuracy_score(o, train_target)

        t = []
        for i in range(len(test_data)):
            my_predict_t = predict_test(i)
            if my_predict_t >= 0:
                t.append(1)
            else:
                t.append(-1)
        my_t = sklearn.metrics.accuracy_score(t, test_target)

        train_accs.append(my)
        test_accs.append(my_t)

        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    print("Training finished after iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
        len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    # TODO: Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).
    # Note that until now the full `a` should have been for prediction.

    support_vectors = []
    support_vector_weights = []

    for i in range(len(a)):
        if a[i] > args.tolerance:
            support_vectors.append(train_data[i])
            support_vector_weights.append((a[i]*train_target[i]))

    return support_vectors, support_vector_weights, b, train_accs, test_accs


def main(args):
    # Generate an artifical regression dataset, with +-1 as targets
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Run the SMO algorithm
    support_vectors, support_vector_weights, bias, train_accs, test_accs = smo(
        args, train_data, train_target, test_data, test_target)

    if args.plot:
        import matplotlib.pyplot as plt
        def plot(predict, support_vectors):
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
            test_mismatch = np.sign([predict(x) for x in test_data]) != test_target
            plt.figure()
            plt.contourf(xs, ys, predictions, levels=0, cmap=plt.cm.RdBu)
            plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap=plt.cm.RdBu,
                        zorder=2)
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker="o", s=90, label="Support Vectors",
                        c="#00dd00")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap=plt.cm.RdBu,
                        zorder=2)
            plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130,
                        label="Test Errors", c="#ffff00")
            plt.legend(loc="upper center", ncol=4)

        # If you want plotting to work (not required for ReCodEx), you need to
        # define `predict_function` computing SVM prediction `y(x)` for the given x.
        predict_function = lambda x: None

        plot(predict_function, support_vectors)
        if args.plot is True:
            plt.show()
        else:
            plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return support_vectors, support_vector_weights, bias, train_accs, test_accs


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
