#!/usr/bin/env python3
import argparse
import time

import numpy as np
from scipy.stats import norm

import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter for Bernoulli and Multinomial NB")
parser.add_argument("--naive_bayes_type", default="gaussian", type=str, help="NB type to use")
parser.add_argument("--classes", default=3, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)


    # TODO: Fit the naive Bayes classifier on the train data.

    class_fitted = []
    class_prob = []

    fitted = np.empty((train_data.shape[1], 2), dtype=float)
    for k in range(len(np.unique(train_target))):
        for i in range(train_data.shape[1]):
            if args.naive_bayes_type == "gaussian":
                mean = np.mean(train_data[train_target == k].T[i])
                var = 0
                for xi in train_data[train_target == k].T[i]:
                    var += (xi - mean) ** 2
                var = var / len(train_data[train_target == k].T[i])
                fitted[i] = [mean, np.sqrt(var + args.alpha)]
            elif args.naive_bayes_type == "multinomial":
                n_total = np.sum(train_data[train_target == k].T.flatten())
                ni = np.sum(train_data[train_target == k].T[i])
                pi = (ni + args.alpha)/(n_total + args.alpha * train_data.shape[1])
                fitted[i] = [999, pi]
            elif args.naive_bayes_type == "bernoulli":
                n_total = train_data[train_target == k].shape[0]
                ni = train_data[train_target == k].T[i]
                ni = len(ni[ni != 0])
                pi = (ni + args.alpha)/(n_total + 2 * args.alpha)
                fitted[i] = [999, pi]
        class_fitted.append(fitted)
        fitted = np.empty((train_data.shape[1], 2), dtype=float)
        class_prob.append(len(train_target[train_target == k]) / len(train_target))

    class_fitted = np.asarray(class_fitted)

    my_test = []
    import time
    t_start = time.time()
    for row in test_data:
        probs = []
        if args.naive_bayes_type == "gaussian":
            probs = np.log(class_prob) + np.sum(norm.logpdf(row, class_fitted[:, :, 0], class_fitted[:, :, 1]),
                                                axis=-1)
        else:
            for k in range(len(np.unique(train_target))):
                p = np.log(class_prob[k])
                for m in range(len(row)):
                    xi = row[m]
                    #if args.naive_bayes_type == "gaussian":
                    #    p_xi_k = norm.logpdf(xi, class_fitted[k][m][0], class_fitted[k][m][1])
                    #    p += p_xi_k
                    if args.naive_bayes_type == "multinomial":
                        p_xi_k = xi * np.log(class_fitted[k][m][1])
                        p += p_xi_k
                    elif args.naive_bayes_type == "bernoulli":
                        if xi > 0:
                            xi = 1
                        p_xi_k = xi * np.log(class_fitted[k][m][1]/(1 - class_fitted[k][m][1])) + np.log(1 - class_fitted[k][m][1])
                        p += p_xi_k
                probs.append(p)
        my_test.append(np.argmax(probs))
    t_end = time.time()
    print("second whole:", t_end - t_start)

    # TODO: Predict the test data classes and compute test accuracy.
    test_accuracy = sklearn.metrics.accuracy_score(my_test, test_target)

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))