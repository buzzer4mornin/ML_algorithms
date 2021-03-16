#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import zipfile

import numpy as np

import sklearn.ensemble
import sklearn.feature_extraction
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.pipeline
import sklearn.svm

class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.zip",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with zipfile.ZipFile(name, "r") as dataset_file:
            with dataset_file.open(os.path.basename(name).replace(".zip", ".txt"), "r") as train_file:
                for line in train_file:
                    label, text = line.decode("utf-8").rstrip("\n").split("\t")
                    self.data.append(text)
                    self.target.append(int(label))
        self.target = np.array(self.target, np.int32)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--cv", default=5, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--features", default="words", type=str, help="Features to use (words/counts/charsN/word_charsN)")
parser.add_argument("--model", default="multinomial_nb", type=str, help="Model type")
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        if args.model == "svm":
            words, estimator = 5000, sklearn.svm.SVC(kernel="rbf", gamma="scale")
        elif args.model == "mlp":
            words, estimator = 2000, sklearn.neural_network.MLPClassifier(verbose=1, hidden_layer_sizes=300, max_iter=3)
        elif args.model == "mlp5":
            words, estimator = 2000, sklearn.ensemble.VotingClassifier(
                [("MLP{}".format(i), sklearn.neural_network.MLPClassifier(verbose=1, hidden_layer_sizes=300, max_iter=3))
                 for i in range(5)], voting="soft")
        elif args.model == "bernoulli_nb":
            words, estimator = 2000, sklearn.naive_bayes.BernoulliNB()
        elif args.model == "multinomial_nb":
            words, estimator = 5000, sklearn.naive_bayes.MultinomialNB(alpha=0.25 if "chars" in args.features else 10)
        elif args.model == "gbt":
            words, estimator = 1000, sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=3, subsample=1, verbose=1)
        elif args.model == "lr":
            words, estimator = 2000, sklearn.linear_model.LogisticRegression(solver="lbfgs", penalty="none", verbose=1)
        else:
            raise ValueError("Unknown model {}".format(args.model))

        if args.features == "words":
            features = sklearn.feature_extraction.text.TfidfVectorizer(strip_accents="unicode", max_features=words, analyzer="word", stop_words="english")
        elif args.features == "counts":
            features = sklearn.feature_extraction.text.CountVectorizer(strip_accents="unicode", max_features=words, analyzer="word", stop_words="english")
        elif args.features.startswith("chars"):
            chars = int(args.features[5:])
            features = sklearn.feature_extraction.text.TfidfVectorizer(analyzer="char", lowercase=False, ngram_range=(1,chars))
        elif args.features.startswith("word_chars"):
            chars = int(args.features[10:])
            features = sklearn.feature_extraction.text.TfidfVectorizer(analyzer="char_wb", lowercase=False, ngram_range=(1,chars))
        else:
            raise ValueError("Unknown features {}".format(args.model))

        # Tried combinations:
        # --model=svm --features=words
        # --model=gbt --features=counts
        # --model=mlp --features=words
        # --model=mlp5 --features=words
        # --model=bernoulli_nb --features=words
        # --model=multinomial_nb --features=words
        # --model=multinomial_nb --features=chars1 (and also chars2, chars3, chars4, chars5)
        # --model=multinomial_nb --features=word_chars1
        #   (and also word_chars2, word_chars3, word_chars4, word_chars5)

        model = sklearn.pipeline.Pipeline([
            ("Features", features),
            ("Estimator", estimator),
        ])

        if args.cv:
            scores = sklearn.model_selection.cross_val_score(model, train.data, train.target, scoring="f1", cv=args.cv)
            print("Cross-validation with {} folds: {:.2f} +-{:.2f}".format(args.cv, 100 * scores.mean(), 100 * scores.std()))

        model.fit(train.data, train.target)

        for mlp in [estimator] if args.model == "mlp" else estimator.estimators_ if args.model == "mlp5" else []:
            mlp._optimizer = None
            for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
            for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list of a NumPy array.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)