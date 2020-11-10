#!/usr/bin/env python3

# Team:
# 2f67b427-a885-11e7-a937-00505601122b
# b030d249-e9cb-11e9-9ce9-00505601122b
# 3351ff04-3f62-11e9-b0fd-00505601122b


import argparse
import lzma
import os
import pickle
import urllib.request
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import warnings
warnings.filterwarnings("ignore")

class Dataset:
    """Thyroid Dataset.
    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features
    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """

    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        X = pd.DataFrame(train.data)
        y = pd.DataFrame(train.target)

        # ==================================================================================================
        'Explanatory Data Analysis'

        # first 15 features are Binary, remaining 6 features are Real-valued
        # for line in range(min(X.shape[0], 5)):
        #    print(" ".join("{:.4g}".format(X.loc[line, column]) for column in range(min(X.shape[1], 60))))

        # print(X.info()) # No NULL values.

        # X = X.iloc[:, 15:21]
        # print(X.describe()) # all real-valued features are in [0,1] interval. So, no need for normalization

        # print(y.iloc[:, 0].value_counts()) # dataset is extremely imbalanced {0:3488, 1:284}

        # X = train.data
        # y = train.target
        # df = pd.DataFrame(np.c_[X, y])
        # corrMatrix = df.corr()
        # sn.heatmap(corrMatrix, annot=True)
        # plt.show()      # 18/20 columns have 0.77 correlation, delete one of them

        # ==================================================================================================
        norm_cols, poly_cols = list(X.columns[15:21]), list(X.columns[0:15])

        col_trans = ColumnTransformer([('norm', StandardScaler(), norm_cols),  # without StandardScale() ~ 0.9841/better
                                       ('poly', PolynomialFeatures(3, include_bias=False), poly_cols)],
                                      remainder='passthrough')

        X = pd.DataFrame(col_trans.fit_transform(X))
        y = pd.DataFrame(np.asarray(y).ravel())

        # Examining Coefficients which is not present on this script..
        # After PolyFeatures, we have 821 features. 29 features out of this 821 have highest impact in predicting output
        frs = [0, 1, 2, 4, 5, 6, 7, 9, 12, 14, 18, 21, 30, 35, 36, 63, 96, 120, 135, 141, 149, 150, 163, 240, 261, 457,
               656, 765, 811]

        # LinearDiscriminantAnalysis on whole 821 features
        lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.7,
                                         store_covariance=True, tol=1.0e-4).fit(X, y)

        # LR with 29 most important features ==> frs
        D_X = X[X.columns[frs]]
        clf_29 = LogisticRegression(random_state=args.seed, solver="liblinear", penalty='l1',
                                    class_weight="balanced", tol=1e-2).fit(D_X, y)

        # LR with 2 most important out of 29 (frs)
        b = [1, 22]
        c = [frs[i] for i in b]
        D_X = X[X.columns[[c]]]
        clf_2 = LogisticRegression(random_state=args.seed, solver="liblinear",
                                   class_weight="balanced", tol=1e-2, penalty='l1').fit(D_X, y)

        # Helper 1 -- 10 randomly sampled features out of 29 important features (frs)
        y_1 = [16, 1, 20, 26, 7, 12, 23, 25, 27, 9]
        c = [frs[i] for i in y_1]
        D_X = X[X.columns[[c]]]
        clf_h_1 = LogisticRegression(random_state=args.seed, solver="liblinear",
                                     class_weight="balanced", tol=1e-2, penalty='l1').fit(D_X, y)

        # Helper 2 -- 10 randomly sampled features out of 29 important features (frs)
        y_2 = [22, 11, 15, 16, 5, 14, 0, 19, 6, 10]
        c = [frs[i] for i in y_2]
        D_X = X[X.columns[[c]]]
        clf_h_2 = LogisticRegression(random_state=args.seed, solver="liblinear",
                                     class_weight="balanced", tol=1e-2, penalty='l1').fit(D_X, y)

        # Helper 3 -- 10 randomly sampled features out of 29 important features (frs)
        y_3 = [1, 12, 8, 15, 3, 16, 9, 5, 25, 4]
        c = [frs[i] for i in y_3]
        D_X = X[X.columns[[c]]]
        clf_h_3 = LogisticRegression(random_state=args.seed, solver="liblinear",
                                     class_weight="balanced", tol=1e-2, penalty='l1').fit(D_X, y)

        # Helper 4 -- 10 randomly sampled features out of 29 important features (frs)
        y_4 = [1, 0, 26, 21, 10, 20, 19, 15, 22, 13]
        c = [frs[i] for i in y_4]
        D_X = X[X.columns[[c]]]
        clf_h_4 = LogisticRegression(random_state=args.seed, solver="liblinear",
                                     class_weight="balanced", tol=1e-2, penalty='l1').fit(D_X, y)

        # Helper 5 -- 10 randomly sampled features out of 29 important features (frs)
        y_5 = [8, 7, 27, 15, 2, 14, 0, 11, 5, 16]
        c = [frs[i] for i in y_5]
        D_X = X[X.columns[[c]]]
        clf_h_5 = LogisticRegression(random_state=args.seed, solver="liblinear",
                                     class_weight="balanced", tol=1e-2, penalty='l1').fit(D_X, y)

        # Helper 6 -- 10 randomly sampled features out of 29 important features (frs)
        y_6 = [7, 6, 16, 18, 15, 8, 0, 3, 5, 14]
        c = [frs[i] for i in y_6]
        D_X = X[X.columns[[c]]]
        clf_h_6 = LogisticRegression(random_state=args.seed, solver="liblinear",
                                     class_weight="balanced", tol=1e-2, penalty='l1').fit(D_X, y)


        """
        Serializing All 9 models 
        """
        with lzma.open("lda.model", "wb") as model_file:
            pickle.dump(lda, model_file)

        with lzma.open("clf_2.model", "wb") as model_file:
            pickle.dump(clf_2, model_file)

        with lzma.open("clf_29.model", "wb") as model_file:
            pickle.dump(clf_29, model_file)

        with lzma.open("clf_h_1.model", "wb") as model_file:
            pickle.dump(clf_h_1, model_file)

        with lzma.open("clf_h_2.model", "wb") as model_file:
            pickle.dump(clf_h_2, model_file)

        with lzma.open("clf_h_3.model", "wb") as model_file:
            pickle.dump(clf_h_3, model_file)

        with lzma.open("clf_h_4.model", "wb") as model_file:
            pickle.dump(clf_h_4, model_file)

        with lzma.open("clf_h_5.model", "wb") as model_file:
            pickle.dump(clf_h_5, model_file)

        with lzma.open("clf_h_6.model", "wb") as model_file:
            pickle.dump(clf_h_6, model_file)

        """
        Serializing Pipeline
        """
        with lzma.open("pipeline.model", "wb") as model_file:
            pickle.dump(col_trans, model_file)


    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)
        X = pd.DataFrame(test.data)

        # Loading all 9 models and pipeline
        with lzma.open("lda.model", "rb") as model_file:
            lda = pickle.load(model_file)

        with lzma.open("clf_2.model", "rb") as model_file:
            clf_2 = pickle.load(model_file)

        with lzma.open("clf_29.model", "rb") as model_file:
            clf_29 = pickle.load(model_file)

        with lzma.open("clf_h_1.model", "rb") as model_file:
            clf_h_1 = pickle.load(model_file)

        with lzma.open("clf_h_2.model", "rb") as model_file:
            clf_h_2 = pickle.load(model_file)

        with lzma.open("clf_h_3.model", "rb") as model_file:
            clf_h_3 = pickle.load(model_file)

        with lzma.open("clf_h_4.model", "rb") as model_file:
            clf_h_4 = pickle.load(model_file)

        with lzma.open("clf_h_5.model", "rb") as model_file:
            clf_h_5 = pickle.load(model_file)

        with lzma.open("clf_h_6.model", "rb") as model_file:
            clf_h_6 = pickle.load(model_file)

        with lzma.open("pipeline.model", "rb") as model_file:
            col_trans = pickle.load(model_file)



        """=========================================================================================================="""
        """================= ENSEMBLE LEARNING -> Comparing results of all 9 models, then deciding output. ====== """
        """=========================================================================================================="""

        # Transform columns according to Pipeline
        norm_cols, poly_cols = list(X.columns[15:21]), list(X.columns[0:15])
        X = pd.DataFrame(col_trans.transform(X))


        frs = [0, 1, 2, 4, 5, 6, 7, 9, 12, 14, 18, 21, 30, 35, 36, 63, 96, 120, 135, 141, 149, 150, 163, 240, 261, 457,
               656, 765, 811]

        b = [1, 22]
        y_1 = [16, 1, 20, 26, 7, 12, 23, 25, 27, 9]
        y_2 = [22, 11, 15, 16, 5, 14, 0, 19, 6, 10]
        y_3 = [1, 12, 8, 15, 3, 16, 9, 5, 25, 4]
        y_4 = [1, 0, 26, 21, 10, 20, 19, 15, 22, 13]
        y_5 = [8, 7, 27, 15, 2, 14, 0, 11, 5, 16]
        y_6 = [7, 6, 16, 18, 15, 8, 0, 3, 5, 14]

        c = [frs[i] for i in y_1]
        tst = X[X.columns[c]]
        y_h_1_pred = clf_h_1.predict(tst)

        c = [frs[i] for i in y_2]
        tst = X[X.columns[c]]
        y_h_2_pred = clf_h_2.predict(tst)

        c = [frs[i] for i in y_3]
        tst = X[X.columns[c]]
        y_h_3_pred = clf_h_3.predict(tst)

        c = [frs[i] for i in y_4]
        tst = X[X.columns[c]]
        y_h_4_pred = clf_h_4.predict(tst)

        c = [frs[i] for i in y_5]
        tst = X[X.columns[c]]
        y_h_5_pred = clf_h_5.predict(tst)

        c = [frs[i] for i in y_6]
        tst = X[X.columns[c]]
        y_h_6_pred = clf_h_6.predict(tst)

        y_lda = lda.predict(X)

        tst = X[X.columns[frs]]
        y_clf_29 = clf_29.predict(tst)

        c = [frs[i] for i in b]
        tst = X[X.columns[c]]
        y_clf_2 = clf_2.predict(tst)

        res = []
        for i in range(len(y_lda)):
            if y_lda[i] + y_clf_29[i] + y_clf_2[i] + y_h_1_pred[i] + y_h_2_pred[i] + y_h_3_pred[i] + \
                    y_h_4_pred[i] + y_h_5_pred[i] + y_h_6_pred[i] >= 5:
                res.append(1)
            else:
                res.append(0)


        predictions = np.array(res)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
