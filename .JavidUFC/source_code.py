# Load necessary libraries
import numpy as np
import pandas as pd
import sklearn.compose
import sklearn.dummy
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import argparse
import lzma
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--cv", default=5, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--model", default="bad_lr", type=str, help="Model to use")
parser.add_argument("--model_path", default="saved_model", type=str, help="Model path")
parser.add_argument("--trees", default=500, type=int, help="Number of trees to use")
parser.add_argument("--max_depth", default=5, type=int, help="Maximum tree depth")
parser.add_argument("--subsample", default=0.7, type=float, help="Subsample data")


def main(args):
    # Read Data
    df = pd.read_csv(r'preprocessed_round_blue.csv')

    # Inspect Data
    # print(df.info())
    # print(df['B_Stance_Orthodox'].value_counts())
    # print(df['avg_SIG_STR_att_diff'].describe())

    # Drop Referee Column
    df = df.drop(columns=['Referee'])

    # Dummy Location Column
    # before: dtypes: float64(47), int64(42), object(2)
    df = pd.get_dummies(df, columns=['location'])
    # after: dtypes: float64(47), int64(42), object(1) --> Output, uint8(26)

    # Additional options
    '''# Select Float Columns
    # float_cols = list(df.loc[:, df.dtypes == np.float64].columns)
    # Correlation HeatMap
    
    # X = train.data
    # y = train.target
    # df = pd.DataFrame(np.c_[X, y])
    # corrMatrix = df.corr()
    # sn.heatmap(corrMatrix, annot=True)
    # plt.show()      # 18/20 columns have 0.77 correlation, delete one of them
    '''

    # Get Output Column
    # Before that Map {Red, Blue} into {1,0} binary.
    df['Winner'] = df['Winner'].map({'Red': 1, 'Blue': 0}).astype(int)
    y = np.array(df['Winner'])

    # Get Input Columns
    # Before that DROP Output Column From Dataframe
    df = df.drop(columns=['Winner'])
    try:
        df = df.drop(columns=['win_by'])
        #df = df.drop(columns=['date'])
    except:
        raise ValueError("Set Correct Dataset")

    X = np.array(df)

    '''if args.model == "lr":
        model = [
            # ("poly", sklearn.preprocessing.PolynomialFeatures(2)),
            ("lr_cv", sklearn.linear_model.LogisticRegressionCV(solver='newton-cg', Cs=np.geomspace(0.001, 1000, 7), max_iter=100)),
        ]
    elif args.model == "adalr":
        model = [
            # ("poly", sklearn.preprocessing.PolynomialFeatures(2)),
            ("ada_lr_cv",
             sklearn.ensemble.AdaBoostClassifier(sklearn.linear_model.LogisticRegression(C=1), n_estimators=50)),
        ]
    elif args.model == "baglr":
        model = [
            # ("poly", sklearn.preprocessing.PolynomialFeatures(2)),
            ("bag_lr_cv",
             sklearn.ensemble.BaggingClassifier(sklearn.linear_model.LogisticRegression(solver='newton-cg', max_iter=100, C=1), n_estimators=50)),
        ]
    elif args.model == "badlr":
        model = [("lr", sklearn.linear_model.LogisticRegression(solver='newton-cg'))]
    elif args.model == "mlp":
        model = [
            ("MLP_ensemble", sklearn.ensemble.VotingClassifier([
                ("MLP{}".format(i), sklearn.neural_network.MLPClassifier(tol=0, learning_rate_init=0.01, max_iter=200,
                                                                         hidden_layer_sizes=(300, 200, 100),
                                                                         activation="relu", solver="adam", verbose=1))
                for i in range(3)])),
        ]'''

    int_columns = np.all(X.astype(int) == X, axis=0)

    model = sklearn.pipeline.Pipeline([
        # ("standard_scaler", sklearn.preprocessing.StandardScaler()),
        ("minmax_scaler", sklearn.preprocessing.MinMaxScaler()),
        ("estimator", {
            "svm": sklearn.svm.SVC(verbose=1),
            "mlp": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(200, 200), solver="adam", learning_rate="constant", max_iter=100, batch_size=50, verbose=1,
                                                        early_stopping=False, activation="relu", validation_fraction=0.15),
            "gbt": sklearn.ensemble.GradientBoostingClassifier(n_estimators=args.trees, max_depth=args.max_depth,
                                                               subsample=args.subsample, verbose=1),
            "rf": sklearn.ensemble.RandomForestClassifier(n_estimators=args.trees, verbose=1),
            "bad_lr": sklearn.linear_model.LogisticRegression(solver='newton-cg'),
            "lr_cv": sklearn.linear_model.LogisticRegressionCV(solver='liblinear', Cs=np.geomspace(0.001, 1000, 7), max_iter=300),
            "ada_lr": sklearn.ensemble.AdaBoostClassifier(sklearn.linear_model.LogisticRegression(solver='liblinear', C=1), n_estimators=10),
            "bag_lr": sklearn.ensemble.BaggingClassifier(sklearn.linear_model.LogisticRegression(solver='liblinear', C=1), n_estimators=10),
        }[args.model]),
    ])

    if args.cv:
        scores = sklearn.model_selection.cross_val_score(model, X, y, cv=args.cv)
        print("Cross-validation with {} folds: {:.2f} +-{:.2f}".format(args.cv, 100 * scores.mean(), 100 * scores.std()))

    #model.fit(X, y)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)