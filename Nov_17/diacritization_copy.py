#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"),
                                       filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")


def main(args):
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()
        data_split = train.data.lower().split()             #TODO: only .lower().split()
        target_split = train.target.lower().split()         #TODO: only .lower().split()


        X_train, X_test, y_train, y_test = train_test_split(data_split, target_split, test_size=0.2,
                                                            random_state=args.seed, shuffle=True)

        X_test_copy = X_test
        X_test = (" ".join(X_test)).lower().split()
        #y_test = (" ".join(y_test)).lower().split()


        y_train = (" ".join(y_train)).lower().split()
        X_train = (" ".join(X_train)).lower().split()


        window_length = len(max(data_split, key=len)) + 1
        reformat_to_window = lambda w: w + ' ' * (window_length - len(w))

        window_length_t = 23
        reformat_to_window_t = lambda w: w + ' ' * (window_length_t - len(w))

        data_formated = [reformat_to_window(w) for w in data_split]      #TODO: X_train data_split
        target_formated = [reformat_to_window_t(w) for w in target_split]  #TODO: y_train  target_split

        switcher = {
            "á": 26,
            "č": 27,
            "ď": 28,
            "é": 29,
            "ě": 30,
            "í": 31,
            "ň": 32,
            "ó": 33,
            "ř": 34,
            "š": 35,
            "ť": 36,
            "ú": 37,
            "ů": 38,
            "ý": 39,
            "ž": 40,
            ",": 41,
            ".": 42,
            "-": 43,
            "!": 44,
            "?": 45,
            '"': 46,
            "(": 47
        }

        transfer = {
            "a": "á",
            "c": "č",
            "d": "ď",
            "e1": "ě",
            "e0": "é",
            "i": "í",
            "n": "ň",
            "o": "ó",
            "r": "ř",
            "s": "š",
            "t": "ť",
            "u1": "ů",
            "u0": "ú",
            "y": "ý",
            "z": "ž"
        }


        alpha = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
        symbols = ["á","č","ď","é","ě","í","ň","ó","ř","š","ť","ú","ů","ý","ž",",",".","-","!","?",'"',"("]
        diacred = ["á","č","ď","é","ě","í","ň","ó","ř","š","ť","ú","ů","ý","ž"]


        '''u1 = False
        u2 = False
        counts = 0
        for i in target_formated:
            for j in i:
                if j == "ú":
                    u1 = True
                if j == "ů":
                    u2 = True
            if u1 == True and u2 == True:
                counts += 1
            u1 = False
            u2 = False

        print("u:",counts)

        e1 = False
        e2 = False
        counts = 0
        for i in target_formated:
            for j in i:
                if j == "é":
                    e1 = True
                if j == "ě":
                    e2 = True
            if e1 == True and e2 == True:
                counts += 1
            e1 = False
            e2 = False

        print("e",counts)'''

        input_default = np.zeros(48, dtype=int)

        def input_one_hot_encoder(l):
            zeros = np.zeros(48, dtype=int)
            zeros[l] = 1
            return zeros

        def encode_window(word):
            word_binaries = [input_one_hot_encoder(ord(l)-97) if (l in alpha and l not in symbols) else input_one_hot_encoder(switcher.get(l)) if l in symbols else input_default for l in list(word)]
            return np.array(word_binaries).flatten()


        def encode_window_output(word):
            indexes = [1 if l in diacred else 0 for l in list(word)]
            u2 = False
            e2 = False
            for j in word:
                if j == "ů":
                    u2 = True
                if j == "ě":
                    e2 = True
            if u2 == True:
                indexes[21] = 1
            if e2 == True:
                indexes[22] = 1
            return np.array(indexes).flatten()


        for i in range(len(data_formated)):
            data_formated[i] = encode_window(data_formated[i])

        for i in range(len(target_formated)):
            target_formated[i] = encode_window_output(target_formated[i])


        network = MLPClassifier(hidden_layer_sizes=(500,), activation='relu', solver='adam', alpha=0.0001,
                                batch_size=300, learning_rate='adaptive', learning_rate_init=0.001, max_iter=140,
                                shuffle=True, random_state=args.seed, tol=1e-5, verbose=True, early_stopping=False)

        model = network.fit(data_formated, target_formated)
        # Serialize the model.

        with lzma.open("diacritization.model", "wb") as model_file:
            pickle.dump(model, model_file)

        '''with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)'''

        def output_decoder(matrix):
            return [1 if x > 0.85 else 0 for x in matrix]

        def getindices(s):
            return [i for i, c in enumerate(s) if c.isupper()]

        #TODO: X_test preparation
        data_formated_x_test = [reformat_to_window(w) for w in X_test]

        for i in range(len(data_formated_x_test)):
            data_formated_x_test[i] = encode_window(data_formated_x_test[i])


        count = 0
        for i in range(len(X_test_copy)):
            predict = model.predict_proba(data_formated_x_test[i].reshape(1, -1))
            predict = output_decoder(np.array(predict[0]))

            final = ""
            for k in range(len(X_test_copy[i])):
                 if predict[k] == 1:
                    letter = X_test[i][k]
                    if letter == "u":
                        if predict[-2] == 1:
                            convert = transfer.get("u1")
                        else:
                            convert = transfer.get("u0")
                    elif letter == "e":
                        if predict[-1] == 1:
                            convert = transfer.get("e1")
                        else:
                            convert = transfer.get("e0")
                    else:
                        convert = transfer.get(letter)

                    if X_test_copy[i][k].isupper():
                        final += convert.upper()
                    else:
                        final += convert
                 else:
                     final += X_test_copy[i][k]

            if (final == y_test[i]):
                print(final, y_test[i])
                count += 1

        print("acc:", count / len(X_test_copy))



    else:
        switcher = {
            "á": 26,
            "č": 27,
            "ď": 28,
            "é": 29,
            "ě": 30,
            "í": 31,
            "ň": 32,
            "ó": 33,
            "ř": 34,
            "š": 35,
            "ť": 36,
            "ú": 37,
            "ů": 38,
            "ý": 39,
            "ž": 40,
            ",": 41,
            ".": 42,
            "-": 43,
            "!": 44,
            "?": 45,
            '"': 46,
            "(": 47
        }

        transfer = {
            "a": "á",
            "c": "č",
            "d": "ď",
            "e1": "ě",
            "e0": "é",
            "i": "í",
            "n": "ň",
            "o": "ó",
            "r": "ř",
            "s": "š",
            "t": "ť",
            "u1": "ů",
            "u0": "ú",
            "y": "ý",
            "z": "ž"
        }


        alpha = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
        symbols = ["á","č","ď","é","ě","í","ň","ó","ř","š","ť","ú","ů","ý","ž",",",".","-","!","?",'"',"("]
        diacred = ["á","č","ď","é","ě","í","ň","ó","ř","š","ť","ú","ů","ý","ž"]

        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)
        test_copy = test.data.split()
        test = test.data.lower().split()

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        def input_one_hot_encoder(l):
            zeros = np.zeros(48, dtype=int)
            zeros[l] = 1
            return zeros

        def encode_window(word):
            word_binaries = [input_one_hot_encoder(ord(l)-97) if (l in alpha and l not in symbols) else input_one_hot_encoder(switcher.get(l)) if l in symbols else input_default for l in list(word)]
            return np.array(word_binaries).flatten()



        window_length = 21
        reformat_to_window = lambda w: w + ' ' * (window_length - len(w))

        data_formated = [reformat_to_window(w) for w in test]
        data_formated_copy = [reformat_to_window(w) for w in test_copy]

        for i in range(len(data_formated)):
            data_formated[i] = encode_window(data_formated[i])

        for i in range(len(data_formated_copy)):
            data_formated_copy[i] = encode_window(data_formated_copy[i])

        def output_decoder(matrix):
            return [1 if x > 0.85 else 0 for x in matrix]


        predictions = []
        for i in range(len(test_copy)):
            predict = model.predict_proba(data_formated[i].reshape(1, -1))
            predict = output_decoder(np.array(predict[0]))

            final = ""
            for k in range(len(test_copy[i])):
                 if predict[k] == 1:
                    letter = test[i][k]
                    if letter == "u":
                        if predict[-2] == 1:
                            convert = transfer.get("u1")
                        else:
                            convert = transfer.get("u0")
                    elif letter == "e":
                        if predict[-1] == 1:
                            convert = transfer.get("e1")
                        else:
                            convert = transfer.get("e0")
                    else:
                        convert = transfer.get(letter)

                    if test_copy[i][k].isupper():
                        final += convert.upper()
                    else:
                        final += convert
                 else:
                     final += test_copy[i][k]
            predictions.append(final)

        predictions = " ".join(predictions)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)