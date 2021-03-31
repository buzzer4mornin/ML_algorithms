#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np
import sklearn.linear_model
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.pipeline

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "Ã¡ÄÄÃ©Ä›Ã­ÅˆÃ³Å™Å¡Å¥ÃºÅ¯Ã½Å¾"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

    TARGET_LETTERS = sorted(set(LETTERS_NODIA + LETTERS_DIA))
    @staticmethod
    def letter_to_target(letter, target_mode):
        if target_mode == "letters":
            return Dataset.TARGET_LETTERS.index(letter)
        elif target_mode == "marks":
            if letter in "Ã¡Ã©Ã­Ã³ÃºÃ½":
                return 1
            if letter in "ÄÄÄ›ÅˆÅ™Å¡Å¥Å¯Å¾":
                return 2
            return 0

    @staticmethod
    def target_to_letter(target, letter, target_mode):
        if target_mode == "letters":
            return Dataset.TARGET_LETTERS[target]
        elif target_mode == "marks":
            if target == 1:
                index = "aeiouy".find(letter)
                return "Ã¡Ã©Ã­Ã³ÃºÃ½"[index] if index >= 0 else letter
            if target == 2:
                index = "cdenrstuz".find(letter)
                return "ÄÄÄ›ÅˆÅ™Å¡Å¥Å¯Å¾"[index] if index >= 0 else letter
            return letter

    def get_features(self, args):
        processed = self.data.lower()
        features, targets, indices = [], [], []
        for i in range(len(processed)):
            if processed[i] not in Dataset.LETTERS_NODIA:
                continue
            features.append([processed[i]])
            for o in range(1, args.window_chars):
                features[-1].append(processed[i - o:i - o + 1])
                features[-1].append(processed[i + o:i + o + 1])
            for s in range(1, args.window_ngrams):
                for o in range(-s, 0+1):
                    features[-1].append(processed[max(i + o, 0):i + o + s + 1])
            targets.append(self.letter_to_target(self.target[i].lower(), args.target_mode))
            indices.append(i)

        return features, targets, indices

class Dictionary:
    def __init__(self,
                 name="fiction-dictionary.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dictionary to `variants`
        self.variants = {}
        with open(name, "r", encoding="utf-8") as dictionary_file:
            for line in dictionary_file:
                nodia_word, *variants = line.rstrip("\n").split()
                self.variants[nodia_word] = variants

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--dictionary", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--dev", default=None, type=float, help="Use given fraction as dev")
parser.add_argument("--hidden_layers", nargs="+", default=[100], type=int, help="Hidden layer sizes")
parser.add_argument("--max_iter", default=100, type=int, help="Max iters")
parser.add_argument("--model", default="lr", type=str, help="Model to use")
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--prune", default=0, type=int, help="Prune features with <= given counts")
parser.add_argument("--solver", default="saga", type=str, help="LR solver")
parser.add_argument("--target_mode", default="marks", type=str, help="Target mode (letters/marks)")
parser.add_argument("--window_chars", default=1, type=int, help="Window characters to use")
parser.add_argument("--window_ngrams", default=4, type=int, help="Window ngrams to use")

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        train_data, train_target, _ = train.get_features(args)
        if args.prune:
            for i in range(len(train_data[0])):
                features = {}
                for data in train_data:
                    features[data[i]] = features.get(data[i], 0) + 1
                for data in train_data:
                    if features[data[i]] <= args.prune:
                        data[i] = "<unk>"

        model = sklearn.pipeline.Pipeline([
            ("one-hot", sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")),
            ("estimator", {
                "lr": sklearn.linear_model.LogisticRegression(solver=args.solver, multi_class="multinomial", max_iter=args.max_iter, verbose=1),
                "mlp": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=args.hidden_layers, max_iter=args.max_iter, verbose=1),
            }[args.model]),
        ])

        if args.dev:
            train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
                train_data, train_target, test_size=args.dev, shuffle=False)

        model.fit(train_data, train_target)

        if args.dev:
            print("Development accuracy: {}%".format(100 * model.score(test_data, test_target)))

        # Compress MLPs
        if args.model == "mlp":
            mlp = model["estimator"]
            mlp._optimizer = None
            for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
            for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        test_data, _, test_indices = test.get_features(args)

        if args.dictionary:
            dictionary = Dictionary()
            test_target, offset = dict(zip(test_indices, model.predict_log_proba(test_data))), 0

            predictions = []
            for line in test.data.split("\n"):
                predictions.append([])

                for word in line.split(" "):
                    if word in dictionary.variants:
                        best_score, best_variant = None, None
                        for variant in dictionary.variants[word]:
                            score = sum(test_target[offset + i][test.letter_to_target(variant[i].lower(), args.target_mode)]
                                        for i in range(len(word)) if offset + i in test_target)
                            if best_score is None or score > best_score:
                                best_score, best_variant = score, variant
                        predictions[-1].append(best_variant)
                    else:
                        predictions[-1].append([])
                        for i in range(len(word)):
                            predictions[-1][-1].append(
                                test.target_to_letter(np.argmax(test_target[offset + i]), word[i].lower(), args.target_mode)
                                if offset + i in test_target
                                else word[i])
                            if word[i].isupper():
                                predictions[-1][-1][-1] = predictions[-1][-1][-1].upper()
                        predictions[-1][-1] = "".join(predictions[-1][-1])
                    offset += len(word) + 1

                predictions[-1] = " ".join(predictions[-1])
            predictions = "\n".join(predictions)
        else:
            test_target = model.predict(test_data)
            predictions = list(test.data)
            for i in range(len(test_target)):
                predictions[test_indices[i]] = test.target_to_letter(test_target[i], test.data[test_indices[i]].lower(), args.target_mode)
                if test.data[test_indices[i]].isupper():
                    predictions[test_indices[i]] = predictions[test_indices[i]].upper()
            predictions = "".join(predictions)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)