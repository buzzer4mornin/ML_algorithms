#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os

import numpy as np

import sklearn.feature_extraction
import sklearn.neural_network
import sklearn.pipeline

class Dataset:
    CLASSES = ["ARA", "DEU", "FRA", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR", "ZHO"]

    def __init__(self, name):
        if not os.path.exists(name):
            raise RuntimeError("The {} was not found, please download it from ReCodEx".format(name))

        # Load the dataset and split it into `data` and `target`.
        self.data, self.prompts, self.levels, self.target = [], [], [], []
        with open(name, "r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                target, prompt, level, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.prompts.append(prompt)
                self.levels.append(level)
                self.target.append(-1 if not target else self.CLASSES.index(target))
        self.target = np.array(self.target, np.int32)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model", default="mlp", type=str, help="Model type")
parser.add_argument("--model_path", default="nli_competition.model", type=str, help="Model path")

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset("nli_dataset.train.txt")
        dev = Dataset("nli_dataset.dev.txt")

        # TODO: Train a model on the given dataset and store it in `model`.
        model = sklearn.pipeline.Pipeline([
            ("feature_extraction",
             sklearn.pipeline.FeatureUnion([
                 ("word_level", sklearn.feature_extraction.text.TfidfVectorizer(
                     lowercase=True, analyzer="word", ngram_range=(1,2), use_idf=False, sublinear_tf=True, max_features=10000)),
                 ("char_level", sklearn.feature_extraction.text.TfidfVectorizer(
                     lowercase=False, analyzer="char", ngram_range=(1,3), use_idf=False, sublinear_tf=True, max_features=10000)),
             ])),
            ("estimator",
             sklearn.neural_network.MLPClassifier(hidden_layer_sizes=150, max_iter=10, verbose=1)),
        ])
        model.fit(train.data, train.target)
        for name, transformer in model["feature_extraction"].transformer_list:
            transformer.stop_words_ = None

        print(model.score(dev.data, dev.target))

        for mlp in [model["estimator"]] if args.model == "mlp" else []:
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