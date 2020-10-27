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
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
import sklearn.compose

class Dataset:

    def __init__(self,
                 name="rental_competition.train.npz",
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
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")



def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        X = train.data
        y = train.target

        # convert to DataFrame
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        col_names = ['season', 'year', 'month', 'hour', 'holiday', 'day_week', 'work_day',
                     'weather', 'temp', 'feel_temp', 'humidity', 'windspeed']
        X.columns = col_names

        # ==================================================================================================
        ''' Notes from HEATMAP
        [8] and [9] have 0.99 correlation, remove one of them. They are both Celcius Temperature column 
        [8],[9] and [0] have 0.35 correlation, because Temperatures are correlated with Seasons
        [0] and [2] have 0.83 correlation, Seasons are correlated with Months 
        '''

        # Drop 'season' column -- because Seasons are highly correlated with Months (~0.83).
        # Seasons are also correlated with Temperatures [8],[9] (~0.35)
        X = X.drop(['season'], axis=1)

        # Drop 'feel_temp' column -- because Temperature is highly correlated with Feeling Temperature (~0.99)
        X = X.drop(['feel_temp'], axis=1)

        # OrdinalEncode 'hour' column. Transfrom from (0,23) interval into (1,24). Reason: get rid of multiplication with 0
        X.loc[:, 'hour'] += 1

        # Same as above, OrdinalEncode 'day_week' column.
        X.loc[:, 'day_week'] += 1

        # Normalize columns [Doesnt Help]
        #X.loc[:, 'month'] = X.loc[:, 'month'] / pd.unique(X.loc[:, 'month']).shape[0]


        '''#One_Hot_Encode
        myencoder = OneHotEncoder(sparse=False)
        month = X.loc[:, ['month']]
        hour = X.loc[:, ['hour']]
        day_week = X.loc[:, ['day_week']]
        weather = X.loc[:, ['weather']]
        X.drop(['month', 'hour', 'day_week', 'weather'], axis=1, inplace=True)
        month = myencoder.fit_transform(month)
        hour = myencoder.fit_transform(hour)
        day_week = myencoder.fit_transform(day_week)
        weather = myencoder.fit_transform(weather)
        X = pd.DataFrame(np.c_[month, hour, day_week, weather, X]) #if remove 'day_week', best_rmse = 57.8
        print(X.shape)'''

        '''# Polynomial Feature
        poly = PolynomialFeatures(3, include_bias=False)
        start_col = X.shape[1]
        X = pd.DataFrame(poly.fit_transform(X))
        X = X.iloc[:, start_col:]'''

        categ_colnames = ['month', 'hour', 'day_week', 'weather']
        col_trans = sklearn.compose.ColumnTransformer([
            ('1hot', sklearn.preprocessing.OneHotEncoder(sparse=False), categ_colnames)])

        poly = PolynomialFeatures(3, include_bias=False)
        pipeline = sklearn.pipeline.Pipeline([('col_trans', col_trans), ('poly', poly)])

        fit = pipeline.fit(X)
        X = fit.transform(X)
        print(X.shape)


        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=args.seed)

        # best_alfa = 0.2
        # best rmse = 58.66
        clf = Lasso(alpha=0.2, tol=0.001)
        clf.fit(X_train, Y_train)

        predicted_Y = clf.predict(X_test)
        rmse = np.math.sqrt(sklearn.metrics.mean_squared_error(predicted_Y, Y_test))
        print(rmse)


        # TODO: Train a model on the given dataset and store it in `model`.
        model = clf

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)


    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)
        D = test.data

        D = pd.DataFrame(D)

        col_names = ['season', 'year', 'month', 'hour', 'holiday', 'day_week', 'work_day',
                     'weather', 'temp', 'feel_temp', 'humidity', 'windspeed']
        D.columns = col_names

        D = D.drop(['season'], axis=1)

        # Drop 'feel_temp' column -- because Temperature is highly correlated with Feeling Temperature (~0.99)
        D = D.drop(['feel_temp'], axis=1)

        # OrdinalEncode 'hour' column. Transfrom from (0,23) interval into (1,24). Reason: get rid of multiplication with 0
        D.loc[:, 'hour'] += 1

        # Same as above, OrdinalEncode 'day_week' column.
        D.loc[:, 'day_week'] += 1

        #One_Hot_Encode
        myencoder = OneHotEncoder(sparse=False)
        month = D.loc[:, ['month']]
        hour = D.loc[:, ['hour']]
        day_week = D.loc[:, ['day_week']]
        weather = D.loc[:, ['weather']]
        D.drop(['month', 'hour', 'day_week', 'weather'], axis=1, inplace=True)
        month = myencoder.fit_transform(month)
        hour = myencoder.fit_transform(hour)
        day_week = myencoder.fit_transform(day_week)
        weather = myencoder.fit_transform(weather)
        D = pd.DataFrame(np.c_[month, hour, day_week, weather, D])

        # Polynomial Feature
        poly = PolynomialFeatures(3, include_bias=False)
        start_col = D.shape[1]
        D = pd.DataFrame(poly.fit_transform(D))
        D = D.iloc[:, start_col:]
        D = np.array(D)


        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        #rmse = np.math.sqrt(sklearn.metrics.mean_squared_error(predicted_Y, y))
        #print(rmse)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = np.array(model.predict(D))

        return predictions

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)