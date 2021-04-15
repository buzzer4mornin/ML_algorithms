# Load necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# Read Data
df = pd.read_csv(r'preprocessed_round.csv')

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
y = np.array(df['Winner'])

# Get Input Columns
# Before that DROP Output Column From Dataframe
df = df.drop(columns=['Winner'])
X = np.array(df)

print(X.shape, y.shape)
exit()


# Get target data
y = df['Winner']
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
label_encoded_y = label_encoder.transform(y)

# Load X variables (IV) into a pandas dataframe with columns
X = df.drop(['Winner'], axis=1)

# Train and test model using Grid Search CV
glm = GridSearchCV(LogisticRegression(), {
    'penalty': ['l1', 'l2'],
    'solver': ['newton-cg', 'liblinear'],
    'max_iter': [100, 150, 200]
}, cv=5, return_train_score=False)

glm.fit(X, y)
df = pd.DataFrame(glm.cv_results_)
print(df[['param_penalty', 'param_solver', 'param_max_iter', 'mean_test_score']])

# Liblinear, 200, L1 is the best

# Train and test model using Randomized Search CV
rlr = RandomizedSearchCV(LogisticRegression(), {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'max_iter': [50, 100, 150, 200]
}, cv=5, return_train_score=False, n_iter=2)
