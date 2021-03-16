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
parser.add_argument("--with_libsvm", default=False, action="store_true", help="Include LibSVM for comparison")

def libsvm(args, train_data, train_target):
    import smo_algorithm
    return smo_algorithm.libsvm(args, train_data, train_target)

def kernel(args, x, y):
    # TODO: Use the kernel from the smo_algorithm assignment.
    import smo_algorithm
    return smo_algorithm.kernel(args, x, y)

def smo(args, train_data, train_target, test_data, test_target):
    # TODO: Use the SMO algorithm from the smo_algorithm assignment.
    import smo_algorithm
    return smo_algorithm.smo(args, train_data, train_target, test_data, test_target)

def main(args):
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    data = sklearn.preprocessing.MinMaxScaler().fit_transform(data)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Using One-vs-One scheme, train (K \binom 2) classifiers, one for every
    # pair of classes $i < j$, using the `smo` method.
    #
    # When training a classifier for classes $i < j$:
    # - keep only the training data of these classes, in the same order
    #   as in the input dataset;
    # - use targets 1 for the class $i$ and -1 for the class $j$.
    test_counts = np.zeros([len(test_data), args.classes], dtype=np.int32)
    for i in range(args.classes):
        for j in range(i + 1, args.classes):
            print("Training classes {} and {}".format(i, j))
            train_indices = (train_target == i) | (train_target == j)
            test_indices = (test_target == i) | (test_target == j)

            support_vectors, support_vector_weights, bias, _, _ = smo(
                args,
                train_data[train_indices], 2 * (train_target[train_indices] == i) - 1,
                test_data[test_indices], 2 * (test_target[test_indices] == i) - 1,
            )

            predictions = bias + sum(w * kernel(args, test_data, s) for s, w in zip(support_vectors, support_vector_weights))
            assert np.all(predictions != 0)

            test_counts[:, i] += predictions > 0
            test_counts[:, j] += predictions < 0

    # TODO: Classify the test set by majority voting of all the trained classifiers,
    # using the lowest class index in the case of ties.
    #
    # Note that during prediction, only the support vectors returned by the `smo`
    # should be used, not all training data.
    #
    # Finally, compute the test set prediction accuracy.
    assert np.all(np.sum(test_counts, axis=1) == args.classes * (args.classes - 1) / 2)
    test_accuracy = sklearn.metrics.accuracy_score(test_target, np.argmax(test_counts, axis=1))

    if args.with_libsvm:
        clf = libsvm(args, train_data, train_target)
        print("LibSVM {:.2f}%".format(100 * sklearn.metrics.accuracy_score(test_target, clf.predict(test_data))))

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))