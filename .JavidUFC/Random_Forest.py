import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Impoort dataset
df = pd.read_csv(r'C:\Users\Asus\Desktop\RSM 2020-2021\BIM Master Thesis\Thesis\Data\Final\Per round data\preprocessed_round.csv')

# Inspect data
df.info()
df
df['Winner'].value_counts()

# Dummy Referee and location
df = pd.get_dummies(df, columns = ['location'])

df = df.drop(columns=['Referee'])

# Get target data
y = df['Winner']
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
label_encoded_y = label_encoder.transform(y)

# Load X variables (IV) into a pandas dataframe with columns
X = df.drop(['Winner'], axis=1)

# Create parameters grid
param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [20, 25, 30], 'max_features': ['auto', 'sqrt'], 'n_estimators': [150, 200, 250]}

# Create a model
rfm = RandomForestClassifier()

# Train and test model using Grid Search CV
rfm_grid = GridSearchCV(estimator = rfm, param_grid = param_grid, cv = 5, return_train_score=False)

rfm_grid.fit(X, y)
df = pd.DataFrame(rfm_grid.cv_results_)
print(df)
print(df[['param_max_features', 'param_criterion', 'param_max_depth', 'param_n_estimators', 'mean_test_score', 'rank_test_score']])