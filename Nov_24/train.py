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
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.

def kernel(x, y):
    # TODO: As in `kernel_linear_regression`, We consider the following `args.kernel`s:
    # - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    return [(args.kernel_gamma * np.dot(x, y) + 1) ** args.kernel_degree if args.kernel == "poly"
                else np.exp(-1 * args.kernel_gamma * ((x - y) @ (x - y)))]




# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.
def smo(args, train_data, train_target, test_data, test_target):
    # Create initial weights

    def train_predict(data_i):
        sum = 0
        for i in range(len(train_data)):
            sum += a[i] * train_target[i] * kernels[data_i, i]
        return sum + b


    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)

    kernels = np.array([[kernel(x, y) for y in train_data] for x in train_data])
    test_kernels = np.array([[kernel(x, y) for y in train_data] for x in test_data])

    passes_without_as_changing = 0
    train_accs, test_accs = [], []
    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # We want j != i, so we "skip" over the value of i
            j = j + (j >= i)

            Ei = train_predict(i) - train_target[i]
            Ej = train_predict(j) - train_target[j]
            # TODO: Check that a[i] fulfils the KKT conditions, using `args.tolerance` during comparisons.
            if ((a[i] < (args.C - args.tolerance)) & (train_target[i] * Ei < -args.tolerance)) or ((a[i] > args.tolerance) & (train_target[i] * Ei > args.tolerance)):
                second_der = 2 * kernels[i, j] - kernels[i, i] - kernels[j, j]
                if second_der > -args.tolerance:
                    continue
                else:  # clipping aj_new
                    aj_new = a[j] - train_target[j] * (Ei - Ej) / second_der
                    # choosing L and H

                    if train_target[i] == train_target[j]:
                        L = max(0, a[i] + a[j] - args.C)
                        H = min(args.C, a[j] + a[i])
                    else:
                        L = max(0, a[j] - a[i])
                        H = min(args.C, args.C + a[j] - a[i])  ### aj or aj_new?

                    aj_new = np.clip(aj_new, L, H)
                    if abs(aj_new - a[j]) < args.tolerance:
                        continue
                    else:  # updating a_i, b
                        ai_new = a[i] - train_target[i] * train_target[j] * (aj_new - a[j])
                        bj_new = b - Ej - train_target[i] * (ai_new - a[i]) * kernels[i, j] - train_target[j] * (aj_new - a[j]) * kernels[j, j]
                        bi_new = b - Ei - train_target[i] * (ai_new - a[i]) * kernels[i, i] - train_target[j] * (aj_new - a[j]) * kernels[j, i]

                        if args.tolerance < ai_new < (args.C - args.tolerance):
                            b_new = bi_new
                        elif args.tolerance < aj_new < (args.C - args.tolerance):
                            b_new = bj_new
                        else:
                            b_new = (bi_new + bj_new) / 2

                        a[j] = aj_new
                        a[i] = ai_new
                        b = b_new

                        as_changed += 1

        # TODO: After each iteration, measure the accuracy for both the
        # train set and the test set and append it to `train_accs` and `test_accs`.
        def test_predict(data_i):
            sum = 0
            for i in range(len(train_data)):
                sum += a[i] * train_target[i] * test_kernels[data_i, i]
            return sum + b

        def train_predict(data_i):
            sum = 0
            for i in range(len(train_data)):
                sum += a[i] * train_target[i] * kernels[data_i, i]
            return sum + b

        kernels = np.array([[kernel(x, y) for y in train_data] for x in train_data])
        test_kernels = np.array([[kernel(x, y) for y in train_data] for x in test_data])

        o = []
        for i in range(100):
            my_predict = train_predict(i)
            if my_predict >= 0:
                o.append(1)
            else:
                o.append(-1)
        my = sklearn.metrics.accuracy_score(o, train_target)


        t = []
        for i in range(len(test_data)):
            my_predict_t = test_predict(i)
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
    support_vectors, support_vector_weights = None, None

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
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker="o", s=90, label="Support Vectors", c="#00dd00")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130, label="Test Errors", c="#ffff00")
            plt.legend(loc="upper center", ncol=4)

        # If you want plotting to work (not required for ReCodEx), you need to
        # define `predict_function` computing SVM prediction `y(x)` for the given x.
        predict_function = lambda x: None

        plot(predict_function, support_vectors)
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return support_vectors, support_vector_weights, bias, train_accs, test_accs

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
