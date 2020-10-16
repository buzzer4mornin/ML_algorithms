#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold


class Dataset:
    """Rental Dataset.
    The dataset instances consist of the following 12 features:
    - [0] season (1: winter, 2: sprint, 3: summer, 4: autumn)                                  [ONE-HOT-ENCODE]
    - [1] year (0: 2011, 1: 2012)                                                              [LEAVE-AS]
    - [2] month (1-12)                                                                         [ONE-HOT-ENCODE]
    - [3] hour (0-23)                                                                          [LEAVE-AS]
    - [4] holiday (binary indicator)                                                           [LEAVE-AS]
    - [5] day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - [6] working day (binary indicator; a day is neither weekend nor holiday)                 [LEAVE-AS]
    - [7] weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - [8] temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - [9] feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - [10] relative humidity (0-1 range)
    - [11] windspeed (normalized to 0-1 range)
    The target variable is the number of rentals in the given hour.
    """
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
        'Explanatory Data Analysis'
        #print(X.head()) # 9 integer features, 3 real features

        #print(X.info()) # No NULL values.

        #X = X.iloc[:,0:5]
        #print(X.describe())

        #X.iloc[:, 5].hist(bins=50, figsize=(15, 5))
        #plt.show()

        #print(X.iloc[:,3].value_counts())

        #X.corrwith(y).plot(kind='barh', figsize=[20, 10], legend=True, grid=True)
        #plt.show()


        #df = pd.DataFrame(np.c_[X,y])
        #print(df.iloc[:,12])
        #corrMatrix = df.corr()
        #print(corrMatrix)
        #sn.heatmap(corrMatrix, annot=True)
        #plt.show()

        #==================================================================================================

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




        # Prepare K-fold cross validation and find average RMSE
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

            reg = LinearRegression()
            reg.fit(train_data, train_target)
            explicit_rmse += np.sqrt(sklearn.metrics.mean_squared_error(reg.predict(test_data), test_target))

        avg_rmse = explicit_rmse/kf.n_splits
        print(avg_rmse)



        '''train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.25,
                                                                            random_state=args.seed)
        reg = LinearRegression()
        reg.fit(train_data, train_target)
        explicit_rmse = np.sqrt(sklearn.metrics.mean_squared_error(reg.predict(test_data), test_target))
        print(explicit_rmse)'''

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