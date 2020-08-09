import copy
import random
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import numpy as np

algorithm_names = ['Adaboost', 'LightGBM', 'Provably Robust Boosting', 'XBART']


def get_best_algorithm(dataset_name):
    random_algorithm = int(random.random() * len(algorithm_names))
    return algorithm_names[random_algorithm]


def read_meta_features():
    df = pd.read_csv('ClassificationAllMetaFeatures.csv')
    df.fillna(df.mean(), inplace=True)
    total_df = pd.DataFrame()
    for index, row in df.iterrows():
        dataset_name = row['dataset']

        best_algorithm = get_best_algorithm(dataset_name)
        for algorithm_index, algorithm in enumerate(algorithm_names):
            current_algorithm_row = row.copy()
            current_algorithm_row['algorithm'] = algorithm_index
            current_algorithm_row['Class'] = int(algorithm == best_algorithm)

            total_df = total_df.append(current_algorithm_row, ignore_index=True)

    X_total = total_df.loc[:, total_df.columns != 'Class']
    Y_total = total_df.loc[:, ['Class']]

    return X_total, Y_total


X_total, Y_total = read_meta_features()
X_no_dataset = X_total.loc[:, X_total.columns != 'dataset']


def train_without_dataset(dataset):
    train_index = X_total['dataset'] != dataset
    test_index = X_total['dataset'] == dataset

    X_train = X_no_dataset[train_index]
    Y_train = Y_total[train_index]
    X_test = X_no_dataset[test_index]
    Y_test = Y_total[test_index]

    xgb_classifier = XGBClassifier()
    xgb_classifier.fit(X_train, Y_train)

    y_pred = xgb_classifier.predict_proba(X_test)
    print(y_pred)
    print(Y_test)
    print('predicted best algorithm:', algorithm_names[int(X_test.iloc[np.argmax(y_pred[:, 1])]['algorithm'])])
    print('real best algorithm:', algorithm_names[int(X_test.iloc[np.argmax(Y_test)]['algorithm'])])


train_without_dataset('iris')
