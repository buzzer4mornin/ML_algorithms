# Load necessary libraries
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--cv", default=5, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--model", default="knn", type=str, help="Model to use")
parser.add_argument("--model_path", default="saved_model", type=str, help="Model path")


def main(args):
    # Read Data
    df = pd.read_csv(r'first_product.csv')

    # Inspect Data
    # print(df.info())
    # print(df['...'].value_counts())
    # print(df['...'].describe())

    # Drop Referee Column
    df = df.drop(columns=['date'])
    df = df.dropna()

    train, test = train_test_split(df, test_size=0.2, random_state=args.seed)

    x_train = train.drop('y', axis=1)
    y_train = train['y']

    x_test = test.drop('y', axis=1)
    y_test = test['y']

    scaler = MinMaxScaler(feature_range=(0, 1))

    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled)

    x_test_scaled = scaler.fit_transform(x_test)
    x_test = pd.DataFrame(x_test_scaled)

    '''rmse_val = []  # to store rmse values for different k
    for K in range(20):
        K = K + 1
        model = neighbors.KNeighborsRegressor(n_neighbors=K)

        model.fit(x_train, y_train)  # fit the model
        pred = model.predict(x_test)  # make prediction on test set
        error = sqrt(mean_squared_error(y_test, pred))  # calculate rmse
        rmse_val.append(error)  # store rmse values
        print('RMSE value for k= ', K, 'is:', error)

    curve = pd.DataFrame(rmse_val)  # elbow curve
    curve.plot()
    plt.show()
    exit()'''

    '''params = {'n_neighbors': range(5, 9), 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'p': [1, 2, 3]}
    knn = neighbors.KNeighborsRegressor()

    model = GridSearchCV(knn, params, cv=5, scoring="neg_root_mean_squared_error")
    model.fit(x_train, y_train)
    print(model.best_params_)'''

    # Best -->  {'algorithm': 'auto', 'leaf_size': 20, 'n_neighbors': 7, 'p': 2, 'weights': 'distance'}

    # Predict on test set
    model = neighbors.KNeighborsRegressor(algorithm="auto", leaf_size=20, n_neighbors=7, p=2, weights="distance")
    model.fit(x_train, y_train)
    y_test_predicted = model.predict(x_test)
    error = sqrt(mean_squared_error(y_test, y_test_predicted))
    print("RMSE:", error)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)