#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.

def kernel(args, x, y):
    return [(args.kernel_gamma * np.dot(x, y) + 1) ** args.kernel_degree if args.kernel == "poly"
            else np.exp(-1 * args.kernel_gamma * ((x - y) @ (x - y)))]

def smo(args, train_data, train_target):
    # Create initial weights

    def predict_train(row):
        sums = 0
        for i in range(len(train_data)):
            sums += a[i] * train_target[i] * train_kernels[row, i]
        return sums + b


    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)
    train_kernels = np.empty((len(train_data), len(train_data)), dtype=float)

    for x1, x1_ in enumerate(train_data):
        for x2, x2_ in enumerate(train_data):
            train_kernels[x1][x2] = kernel(args, x1_, x2_)[0]


    passes_without_as_changing = 0

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


        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

    return a, b

def main(args):
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    data = sklearn.preprocessing.MinMaxScaler().fit_transform(data)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = [], [], [], [], [], [], [], [], [], []

    for i in range(len(train_data)):
        if train_target[i] == 0:
            c0.append(train_data[i])
        elif train_target[i] == 1:
            c1.append(train_data[i])
        elif train_target[i] == 2:
            c2.append(train_data[i])
        elif train_target[i] == 3:
            c3.append(train_data[i])
        elif train_target[i] == 4:
            c4.append(train_data[i])
        elif train_target[i] == 5:
            c5.append(train_data[i])
        elif train_target[i] == 6:
            c6.append(train_data[i])
        elif train_target[i] == 7:
            c7.append(train_data[i])
        elif train_target[i] == 8:
            c8.append(train_data[i])
        else:
            c9.append(train_data[i])

    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = np.array(c0), np.array(c1), np.array(c2), np.array(c3), np.array(c4), np.array(c5), np.array(c6), np.array(c7), np.array(c8), np.array(c9)
    c0_, c1_, c2_, c3_, c4_, c5_, c6_, c7_, c8_, c9_ = np.full(len(c0), 0), np.full(len(c1), 1), np.full(len(c2), 2), np.full(len(c3), 3), \
                        np.full(len(c4), 4), np.full(len(c5), 5), np.full(len(c6), 6), np.full(len(c7), 7), \
                        np.full(len(c8), 8),np.full(len(c9), 9)

    for i in range(10):
        for j in range(10):
            if j <= i:
                continue
            print(i, j)




    # TODO: Using One-vs-One scheme, train (K \binom 2) classifiers, one for every
    # pair of classes $i < j$, using the `smo` method.
    #
    # When training a classifier for classes $i < j$:
    # - keep only the training data of these classes, in the same order
    #   as in the input dataset;
    # - use targets 1 for the class $i$ and -1 for the class $j$.




    # TODO: Classify the test set by majority voting of all the trained classifiers,
    # using the lowest class index in the case of ties.
    #
    # Finally, compute the test set prediction accuracy.
    test_accuracy = None

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    #print("Test set accuracy: {:.2f}%".format(100 * accuracy))
