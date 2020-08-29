import os
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

algorithm_names = ['Adaboost', 'LightGBM', 'Provably Robust Boosting', 'XBART']


def get_best_algorithm(dataset_name, results_directory):
    df = pd.read_csv(results_directory + '/' + dataset_name + '.csv')
    a = df.loc[:, ['Algorithm Name', 'AUC']].groupby('Algorithm Name').mean()
    max_idx = a.idxmax()
    max_auc = a['AUC'].max()
    return list(a[a['AUC'] == max_auc].index.values)
    # random_algorithm = int(random.random() * len(algorithm_names))
    # return algorithm_names[random_algorithm]


def read_meta_features(results_directory):
    df = pd.read_csv('ClassificationAllMetaFeatures.csv')
    df.fillna(df.mean(), inplace=True)
    total_df = pd.DataFrame()
    for index, row in df.iterrows():
        dataset_name = row['dataset']
        if dataset_name + '.csv' in os.listdir(results_directory):
            best_algorithms = get_best_algorithm(dataset_name, results_directory)
            for algorithm_index, algorithm in enumerate(algorithm_names):
                current_algorithm_row = row.copy()
                current_algorithm_row['algorithm'] = algorithm_index
                current_algorithm_row['Class'] = int(algorithm in best_algorithms)

                total_df = total_df.append(current_algorithm_row, ignore_index=True)
        else:
            print("missing results file for", dataset_name)

    X_total = total_df.loc[:, total_df.columns != 'Class']
    Y_total = total_df.loc[:, ['Class']]
    X_no_dataset = X_total.loc[:, X_total.columns != 'dataset']

    return X_total, Y_total, X_no_dataset


def acc4certain_dataset(dataset, X_total, Y_total, X_no_dataset):
    train_index = X_total['dataset'] != dataset
    test_index = X_total['dataset'] == dataset

    X_train = X_no_dataset[train_index]
    Y_train = Y_total[train_index]
    X_test = X_no_dataset[test_index]
    Y_test = Y_total[test_index]

    xgb_classifier = XGBClassifier(learning_rate=0.75)
    xgb_classifier.fit(X_train, Y_train['Class'].ravel())

    y_pred = xgb_classifier.predict(X_test)
    return accuracy_score(Y_test.values, y_pred)


def train_on_all_datasets(X_no_dataset, Y_total):
    X_train = X_no_dataset
    Y_train = Y_total
    xgb_classifier = XGBClassifier(learning_rate=0.75)
    xgb_classifier.fit(X_train, Y_train['Class'].ravel())

    return xgb_classifier
