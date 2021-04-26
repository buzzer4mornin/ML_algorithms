import numpy as np

# TODO ---- Read Data ----
# df = pd.read_csv(r'preprocessed_round_blue.csv')


# TODO ---- Split Train/Test ----
# train_data, test_data, train_target, test_target =
# sklearn.model_selection.train_test_split(data, target, test_size=args.test_size, random_state=args.seed)


# TODO ---- Inspect Data ----
# print(df.info())
# print(df['col_name'].value_counts())
# print(df['col_name'].describe())


# TODO ---- Drop Column ----
# df = df.drop(columns=['Referee'])


# TODO ---- Select Float Columns ----
# float_cols = list(df.loc[:, df.dtypes == np.float64].columns)     <-- from DataFrame
# int_cols = np.all(train_data.astype(int) == train_data, axis=0)   <-- from NumPy


# TODO ---- Correlation HeatMap ----
# X = train.data
# y = train.target
# df = pd.DataFrame(np.c_[X, y])
# corrMatrix = df.corr()
# sn.heatmap(corrMatrix, annot=True)
# plt.show()


# TODO ---- Mapping Binary Strings into 0/1 ----
# df['Winner'] = df['Winner'].map({'Red': 1, 'Blue': 0}).astype(int)


# TODO ---- Dropping Necessary Columns -----
# def get_null_columns(df, threshold=0.9):
#     return df.columns[(df.isnull().sum() / df.shape[0]) > 0.9]
#
# def get_big_top_value_columns(df, threshold=0.9):
#     return [col for col in df.columns if
#                                 df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
# train_null_columns = get_null_columns(train)
# test_null_columns = get_null_columns(test)
# train_big_val_columns = get_big_top_value_columns(train)
# test_big_val_columns = get_big_top_value_columns(test)
#
# cols_to_drop = set(list(train_null_columns) + list(test_null_columns) + train_big_val_columns + test_big_val_columns)
#
# train.drop(cols_to_drop, axis=1, inplace=True)
# test.drop(cols_to_drop, axis=1, inplace=True)


# TODO ---- Get High Correlation Columns ----
# def get_high_correlation_cols(df, corrThresh=0.9):
#     numeric_cols = df._get_numeric_data().columns
#     corr_matrix = df.loc[:, numeric_cols].corr().abs()
#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
#     to_drop = [column for column in upper.columns if any(upper[column] > corrThresh)]
#     return to_drop


# TODO ---- Replace NA ----
# def replace_na(data, numeric_replace=-1, categorical_replace='missing', cat_features=[]):
#     numeric_cols = data._get_numeric_data().columns
#     categorical_cols = list(set(list(set(data.columns) - set(numeric_cols)) + cat_features))
#     categorical_cols = [col for col in categorical_cols if col in data.columns]
#     if numeric_replace is not None:
#         data[numeric_cols] = data[numeric_cols].fillna(numeric_replace)
#     data[categorical_cols] = data[categorical_cols].fillna(categorical_replace)
#     return data
