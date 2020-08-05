import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from WrappedModels.wrapper_provably_robust_boosting import Wrapper_provably_robust_boosting as wprb
from scipy.stats import uniform, randint, rv_discrete
from sklearn.metrics import accuracy_score, make_scorer, precision_score, roc_curve, auc, precision_recall_curve, \
    roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
import lightgbm
from xbart import XBART
from WrappedModels.wrapper_xbart import Wrapper_xbart as wxbart
import time
from sklearn.preprocessing import LabelEncoder
import numpy as np

df_columns = ['Dataset Name', 'Algorithm Name', 'Cross Validation[1-10]', 'Hyper-Parameters Values', 'Accuracy', 'TPR',
              'FPR', 'Precision', 'AUC', 'PR Curve', 'Training Time', 'Inference Time']
df_results = pd.DataFrame(columns=df_columns)
data_dict = {}

directory = 'classification_datasets'
models = []

num_leaves_k = [5, 10, 50, 100]
num_leaves_dist = rv_discrete(name='num_leaves_dist',
                              values=(num_leaves_k, [1 / len(num_leaves_k)] * len(num_leaves_k)))
n_estimators_k = [20, 50, 100, 200]
n_estimators_dist = rv_discrete(name='n_estimators_dist',
                                values=(n_estimators_k, [1 / len(n_estimators_k)] * len(n_estimators_k)))
light_dist_dict = dict(num_leaves=num_leaves_dist, n_estimators=n_estimators_dist)
models.append(('LightGBM', lightgbm.LGBMClassifier, lightgbm.LGBMClassifier(), light_dist_dict))

xk = [5, 10, 50, 100, 200, 500, 1000]
pk = [1 / len(xk)] * len(xk)
min_samples_split_dist = rv_discrete(name='min_samples_split_dist', values=(xk, pk))
wprb_dist_dict = dict(estimator__min_samples_split=min_samples_split_dist, estimator__max_depth=randint(3, 6))
# models.append(('Provably Robust Boosting', wprb, OneVsRestClassifier(wprb()), wprb_dist_dict))

num_trees_k = [20, 50, 100, 200]
num_trees_dist = rv_discrete(name='num_trees_dist', values=(num_trees_k, [1 / len(num_trees_k)] * len(num_trees_k)))
num_sweeps_k = [20, 40, 60, 80]
num_sweeps_dist = rv_discrete(name='num_sweeps_dist',
                              values=(num_sweeps_k, [1 / len(num_sweeps_k)] * len(num_sweeps_k)))
wxbart_dist_dict = dict(num_trees=num_trees_k, num_sweeps=num_sweeps_dist)
# models.append(('XBART', wxbart, wxbart(), wxbart_dist_dict))

random_state = 42
external_split = 2
internal_split = 2
optimization_iterations = 2


def fix_dataset(df):
    le = LabelEncoder()
    for i in range(len(df.columns)):
        if df.iloc[:, i].dtype.name in ['category', 'object']:
            df.iloc[:, i].fillna(df.iloc[:, i].mode().iloc[0], inplace=True)
            df.iloc[:, i] = le.fit_transform(df.iloc[:, i])
        elif 'int' in df.iloc[:, i].dtype.name:
            df.iloc[:, i].fillna(round(df.iloc[:, i].mean()), inplace=True)
        else:
            df.iloc[:, i].fillna(df.iloc[:, i].mean(), inplace=True)
    df.columns = [x for x in range(len(df.columns[:-1]))] + ['Class']
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1:]
    return X, Y


for filename in os.listdir(directory):
    # filename = 'iris.csv'
    # filename = 'lupus.csv'
    # filename = 'kidney.csv'
    # filename = 'autos.csv'
    print(filename)
    data_dict['Dataset Name'] = filename.replace('.csv', '')
    df = pd.read_csv(directory + '/' + filename)
    X, Y = fix_dataset(df)
    kf = KFold(n_splits=external_split, random_state=random_state, shuffle=True)
    for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
        data_dict['Cross Validation[1-10]'] = fold_index
        print("fold index =", fold_index)
        x_train = X.iloc[train_index]
        y_train = Y.iloc[train_index]
        x_test = X.iloc[test_index]
        y_test = Y.iloc[test_index]
        for model_name, model_class, model, model_dict in models:
            print('Model:', model_name)
            data_dict['Algorithm Name'] = model_name
            # distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
            distributions = model_dict
            start_training_time = time.time()
            randomSearcher = RandomizedSearchCV(model, distributions, random_state=random_state, cv=internal_split,
                                                n_iter=optimization_iterations, scoring=make_scorer(accuracy_score))
            randomSearcher.fit(x_train, y_train.values.ravel())

            if model_class is wprb:
                params = {k.replace("estimator__", ""): v for k, v in randomSearcher.best_params_.items()}
                best_model = OneVsRestClassifier(model_class(**params))
            else:
                params = randomSearcher.best_params_
                best_model = model_class(**params)
            data_dict['Hyper-Parameters Values'] = params
            best_model.fit(x_train, y_train.values.ravel())
            data_dict['Training Time'] = time.time() - start_training_time
            print("best params:", best_model)
            print("train accuracy:", accuracy_score(y_train, best_model.predict(x_train)))
            start_inference_time = time.time()
            test_pred = best_model.predict(x_test)
            test_pred_proba = best_model.predict_proba(x_test)
            data_dict['Inference Time'] = (time.time() - start_inference_time) / (len(x_test)) * 1000
            print("test accuracy:", accuracy_score(y_test, test_pred))
            print()
            data_dict['Accuracy'] = accuracy_score(y_test, test_pred)
            data_dict['Precision'] = precision_score(y_test, test_pred, average='macro')
            unique_labels = np.unique(Y.values)
            if len(unique_labels) == 2:  # multiclass vs binary classification
                data_dict['AUC'] = roc_auc_score(y_true=y_test, y_score=test_pred_proba[:, 1])
            else:
                data_dict['AUC'] = roc_auc_score(y_true=y_test, y_score=test_pred_proba, multi_class='ovr',
                                                 labels=np.unique(Y.values))
            # check
            # fpr, tpr, _ = roc_curve(y_test, test_pred)
            data_dict['PR Curve'] = 0
            data_dict['FPR'] = 0
            data_dict['TPR'] = 0
            data_dict['AUC'] = 0
            # data_dict['FPR'] = np.mean(fpr)
            # data_dict['TPR'] = np.mean(tpr)
            # data_dict['AUC'] = auc(fpr, tpr)
            # data_dict['AUC'] = roc_auc_score(y_test, test_pred_proba, multi_class="ovo")
            #
            # precision, recall, thresholds = precision_recall_curve(y_test, test_pred_proba)
            # data_dict['PR Curve'] = auc(precision, recall)
            df_results = df_results.append(data_dict, ignore_index=True)
    df_results.to_csv('Results/' + filename)
    df_results = df_results.iloc[0:0]
