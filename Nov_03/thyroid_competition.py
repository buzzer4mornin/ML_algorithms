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
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

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

        X = col_trans.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=args.seed, shuffle=True)

        y_train = np.asarray(y_train).ravel()
        y_test = np.asarray(y_test).ravel()

        # =================================== Logistic Regression =======================================
        clf = LogisticRegression(random_state=args.seed, solver="liblinear",
                                 class_weight="balanced", tol=1e-2, penalty='l1').fit(X_train, y_train)

        predicted_Y_lr = clf.predict(X_test)
        count = 0
        for i, j in zip(predicted_Y_lr, y_test):
            if i == j: count += 1
        print("Logistic Regression:", count / y_test.shape[0])

        # =============================== Linear Discriminant Analysis ==================================
        # solver='lsqr', shrinkage=0.7 --> [0.940]
        # solver='eigen',shrinkage=0.7 --> [0.940]
        lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.7,
                                         store_covariance=True, tol=1.0e-4).fit(X_train, y_train)

        predicted_Y_lda = lda.predict(X_test)
        count = 0
        for i, j in zip(predicted_Y_lda, y_test):
            if i == j: count += 1
        print("Linear DiscriminantAnalysis:", count / y_test.shape[0])
        # ================================= Support Vector Machine ======================================
        #svm = LinearSVC(loss='squared_hinge', penalty='l2',class_weight='balanced', tol=1e-4,
        #                max_iter=4000, random_state=args.seed).fit(X_train, y_train)
        svm = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale',
                  class_weight='balanced', max_iter=-1, random_state=args.seed).fit(X_train, y_train)

        predicted_Y_svc = svm.predict(X_test)
        count = 0
        for i, j in zip(predicted_Y_svc, y_test):
            if i == j: count += 1
        print("Support Vector Machines:", count / y_test.shape[0])
        # ================================== LDA vs LR comparison =======================================
        count = 0
        for i, j, k in zip(predicted_Y_lr, predicted_Y_lda, y_test):
            if i != k and j == k: count += 1
        print("=====Comparisons=====\nLDA improvement is", count, "more than LR out of", y_test.shape[0])
        # ================================== SVM vs LR comparison =======================================
        count = 0
        for i, j, k in zip(predicted_Y_lr, predicted_Y_svc, y_test):
            if i != k and j == k: count += 1
        print("SVM improvement is", count, "more than LR out of", y_test.shape[0])
        # ================================== SVM vs LDA vs LR comparison =======================================
        count = 0
        for i, j, k, m in zip(predicted_Y_lr, predicted_Y_lda, predicted_Y_svc, y_test):
            if i != m and j == m and k == m: count += 1
            #if i != k and j == k: count += 1
        print("SVM + LDA", count, "more than LR out of", y_test.shape[0])
        # ==================================================================================================

        '''# Prepare K-fold cross validation and find average RMSE
        X = np.asarray(X)
        y = np.asarray(y)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        all_pairs = kf.split(X)

        explicit_rmse = 0
        for train_indices, test_indices in all_pairs:
            train_data = X[train_indices]
            test_data = X[test_indices]
            train_target = y[train_indices]
            test_target = y[test_indices]

            train_target = np.asarray(train_target).ravel()
            test_target = np.asarray(test_target).ravel()
            clf = LogisticRegression(random_state=args.seed, solver="liblinear", penalty='l1',
                                     class_weight="balanced", tol=1e-2).fit(train_data, train_target)
            predicted_Y = clf.predict(test_data)
            count = 0
            for i, j in zip(predicted_Y, test_target):
                if i == j: count += 1

            explicit_rmse += count / test_target.shape[0]

        avg_rmse = explicit_rmse / kf.n_splits
        print(avg_rmse)'''

        # ==================================================================================================

        # TODO: Train a model on the given dataset and store it in `model`.
        model = None

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = None

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
