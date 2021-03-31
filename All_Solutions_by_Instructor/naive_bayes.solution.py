#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.naive_bayes

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter for Bernoulli and Multinomial NB")
parser.add_argument("--naive_bayes_type", default="gaussian", type=str, help="NB type to use")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--with_reference", default=False, action="store_true", help="Show also reference results")

def main(args):
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Train a naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    #
    #   During prediction, you can compute probability density function of a Gaussian
    #   distribution using `scipy.stats.norm`, which offers `pdf` and `logpdf`
    #   methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Do not forget that Bernoulli NB works with binary data, so consider
    #   all non-zero features as ones during both estimation and prediction.
    # Fit priors
    priors = np.bincount(train_target) / len(train_target)

    # Fit the parameters
    params = np.zeros((train_data.shape[1], args.classes, 2))
    for c in range(args.classes):
        c_data = train_data[train_target == c]
        if args.naive_bayes_type == "gaussian":
            params[:, c, 0] = np.mean(c_data, axis=0)
            params[:, c, 1] = np.sqrt(np.var(c_data, axis=0) + args.alpha)
        if args.naive_bayes_type == "multinomial":
            params[:, c, 0] = np.sum(c_data, axis=0) + args.alpha
            params[:, c, 0] = np.log(params[:, c, 0] / np.sum(params[:, c, 0]))
        if args.naive_bayes_type == "bernoulli":
            params[:, c, 0] = (np.sum(c_data != 0, axis=0) + args.alpha) / (len(c_data) + 2 * args.alpha)

    # Model prediction
    log_probabilities = np.zeros((len(test_data), args.classes))
    log_probabilities += np.log(priors)
    if args.naive_bayes_type == "gaussian":
        log_probabilities += np.sum(scipy.stats.norm(loc=params[:, :, 0], scale=params[:, :, 1]).logpdf(np.expand_dims(test_data, -1)), axis=1)
    if args.naive_bayes_type == "multinomial":
        log_probabilities += np.sum(np.expand_dims(test_data, -1) * params[:, :, 0], axis=1)
    if args.naive_bayes_type == "bernoulli":
        log_probabilities += np.sum(np.expand_dims(test_data != 0, -1) * np.log(params[:, :, 0]), axis=1)
        log_probabilities += np.sum(np.expand_dims(test_data == 0, -1) * np.log(1 - params[:, :, 0]), axis=1)

    # TODO: Predict the test data classes and compute test accuracy.
    test_accuracy = sklearn.metrics.accuracy_score(test_target, np.argmax(log_probabilities, axis=1))

    if args.with_reference:
        if args.naive_bayes_type == "gaussian":
            nb = sklearn.naive_bayes.GaussianNB(var_smoothing=args.alpha/np.var(train_data, axis=0).max())
        if args.naive_bayes_type == "multinomial":
            nb = sklearn.naive_bayes.MultinomialNB(alpha=args.alpha)
        if args.naive_bayes_type == "bernoulli":
            nb = sklearn.naive_bayes.BernoulliNB(alpha=args.alpha)
        nb.fit(train_data, train_target)
        test_accuracy = sklearn.metrics.accuracy_score(test_target, nb.predict(test_data))
        print("Scikit-learn test accuracy {:.2f}%".format(
            100 * sklearn.metrics.accuracy_score(test_target, nb.predict(test_data))))

    if args.plot:
        import matplotlib.cm
        import matplotlib.pyplot as plt
        def plot(data, title, mappable=None):
            plt.subplot(3, 1, 1 + len(plt.gcf().get_axes()) // 2)
            plt.imshow(data, cmap="plasma", interpolation="none")
            plt.axis("off")
            plt.colorbar(mappable, aspect=10)
            plt.title(title)

        H = np.sqrt(params.shape[0]).astype(np.int)
        data = np.exp(params) if args.naive_bayes_type == "multinomial" else params
        data = np.pad(data.reshape([H, H, -1, 2]).transpose([3, 0, 2, 1]).reshape([2, H, -1]), [(0, 0), (0, 0), (0, H * (10 - args.classes))])
        plt.figure(figsize=(8*1, 0.9*3))
        plot(data[0], "Estimated means" if args.naive_bayes_type == "gaussian" else "Estimated probabilities")
        if args.naive_bayes_type == "gaussian":
            plot(data[1], "Estimated standard deviations")
            combined = matplotlib.cm.plasma(data[0] / np.max(data[0]))
            combined[:, :, 1] = data[1] / np.max(data[1])
            plot(combined, "Estimated means (R+B) and stds (G)", plt.gcf().get_axes()[0].get_images()[0])
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight", pad_inches=0)

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))