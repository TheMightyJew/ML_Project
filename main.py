import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from WrappedModels.wrapper_provably_robust_boosting import Wrapper_provably_robust_boosting as wprb
from scipy.stats import uniform, randint, rv_discrete
from sklearn.metrics import accuracy_score, make_scorer, precision_score, auc, precision_recall_curve, \
    roc_auc_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
import lightgbm
from WrappedModels.wrapper_xbart import Wrapper_xbart as wxbart
import time
from sklearn.preprocessing import LabelEncoder
import numpy as np
from meta_learner import acc4certain_dataset, read_meta_features, train_on_all_datasets
from xgboost import plot_importance

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

df_columns = ['Dataset Name', 'Algorithm Name', 'Cross Validation[1-10]', 'Hyper-Parameters Values', 'Accuracy', 'TPR',
              'FPR', 'Precision', 'AUC', 'PR Curve', 'Training Time', 'Inference Time']
df_results = pd.DataFrame(columns=df_columns)
data_dict = {}

directory = 'classification_datasets'

n_estimators_k = [20, 50, 100, 200]
n_estimators_dist = rv_discrete(name='n_estimators_dist',
                                values=(n_estimators_k, [1 / len(n_estimators_k)] * len(n_estimators_k)))
adb_dist_dict = dict(learning_rate=uniform(loc=0.1, scale=1.9), n_estimators=n_estimators_dist)

num_leaves_k = [5, 10, 50, 100]
num_leaves_dist = rv_discrete(name='num_leaves_dist',
                              values=(num_leaves_k, [1 / len(num_leaves_k)] * len(num_leaves_k)))
light_dist_dict = dict(num_leaves=num_leaves_dist, n_estimators=n_estimators_dist)

min_samples_split_k = [20, 50, 100, 200, 500, 1000, 2000]
min_samples_split_prob = [1 / len(min_samples_split_k)] * len(min_samples_split_k)
min_samples_split_dist = rv_discrete(name='min_samples_split_dist',
                                     values=(min_samples_split_k, min_samples_split_prob))
min_samples_leaf_dist = rv_discrete(name='min_samples_split_dist',
                                    values=([sample * 0.5 for sample in min_samples_split_k], min_samples_split_prob))
wprb_dist_dict = dict(estimator__min_samples_split=min_samples_split_dist,
                      estimator__min_samples_leaf=min_samples_leaf_dist, estimator__max_depth=randint(2, 5))

num_trees_k = [20, 50, 100, 200]
num_trees_dist = rv_discrete(name='num_trees_dist', values=(num_trees_k, [1 / len(num_trees_k)] * len(num_trees_k)))
num_sweeps_k = [20, 40, 60, 80]
num_sweeps_dist = rv_discrete(name='num_sweeps_dist',
                              values=(num_sweeps_k, [1 / len(num_sweeps_k)] * len(num_sweeps_k)))
burnin_k = [0, 1, 2, 5, 10, 15]
burnin_dist = rv_discrete(name='burnin_dist',
                          values=(burnin_k, [1 / len(burnin_k)] * len(burnin_k)))
max_depth_num_k = [3, 4, 5, 6]
max_depth_num_dist = rv_discrete(name='max_depth_num_dist',
                                 values=(max_depth_num_k, [1 / len(max_depth_num_k)] * len(max_depth_num_k)))
wxbart_dist_dict = dict(num_trees=num_trees_k, num_sweeps=num_sweeps_dist, burnin=burnin_dist,
                        max_depth_num=max_depth_num_dist, alpha=uniform(loc=0.1, scale=1.9),
                        beta=uniform(loc=0.1, scale=1.9))

algorithms_dict = {}
algorithms_dict['Adaboost'] = ('Adaboost', AdaBoostClassifier, AdaBoostClassifier(), adb_dist_dict)
algorithms_dict['LightGBM'] = ('LightGBM', lightgbm.LGBMClassifier, lightgbm.LGBMClassifier(), light_dist_dict)
algorithms_dict['Provably Robust Boosting'] = (
    'Provably Robust Boosting', wprb, OneVsRestClassifier(wprb()), wprb_dist_dict)
algorithms_dict['XBART'] = ('XBART', wxbart, wxbart(), wxbart_dist_dict)


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

    changed = True
    while changed:
        original_class_labels = np.unique(Y.values)
        changed = False
        for class_index in range(len(original_class_labels)):
            if Y.iloc[:, 0].value_counts()[original_class_labels[class_index]] < external_split:
                changed = True
                if class_index > 0:
                    before = Y.iloc[:, 0].value_counts()[original_class_labels[class_index - 1]]
                else:
                    before = 0
                if class_index < len(original_class_labels) - 1:
                    after = Y.iloc[:, 0].value_counts()[original_class_labels[class_index + 1]]
                else:
                    after = 0
                if 0 < after < before:
                    change_to = original_class_labels[class_index + 1]
                else:
                    change_to = original_class_labels[class_index - 1]
                Y = Y.replace(original_class_labels[class_index], change_to)
                break

    Y['Class'] = le.fit_transform(Y['Class'])
    return X, Y


def run_test(filename, results_dir, models, random_state, external_split, internal_split, optimization_iterations):
    global df_results
    print(filename)
    data_dict['Dataset Name'] = filename.replace('.csv', '')
    df = pd.read_csv(directory + '/' + filename)
    X, Y = fix_dataset(df)
    kf = StratifiedKFold(n_splits=external_split, random_state=random_state, shuffle=True)
    for fold_index, (train_index, test_index) in enumerate(kf.split(X, Y)):
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
            print("best params:", params)
            print("train accuracy:", round(accuracy_score(y_train, best_model.predict(x_train)), 4))
            start_inference_time = time.time()
            test_pred = best_model.predict(x_test)
            test_pred_proba = best_model.predict_proba(x_test)
            data_dict['Inference Time'] = (time.time() - start_inference_time) / (len(x_test)) * 1000
            print("test accuracy:", round(accuracy_score(y_test, test_pred), 4))
            print()
            data_dict['Accuracy'] = accuracy_score(y_test, test_pred)
            data_dict['Precision'] = precision_score(y_test, test_pred, average='macro', labels=np.unique(test_pred))
            unique_labels = np.unique(Y.values)
            if len(unique_labels) == 2:  # multiclass vs binary classification
                data_dict['AUC'] = roc_auc_score(y_true=y_test, y_score=test_pred_proba[:, 1])
            else:
                # plaster = test_pred_proba[:, [np.where(np.unique(Y.values) == x)[0][0] for x in np.unique(y_test)]]
                # plaster2 = np.array([[x / sum(y) for x in y] for y in plaster])
                data_dict['AUC'] = roc_auc_score(y_true=y_test, y_score=test_pred_proba, multi_class='ovr',
                                                 labels=np.unique(y_test))
            all_TPR = []
            all_FPR = []
            all_PR_CURVE = []
            for index, class_label in enumerate(np.unique(y_test)):
                tn, fp, fn, tp = confusion_matrix(y_test == class_label, test_pred == class_label).ravel()
                all_FPR.append(fp / (fp + tn))
                all_TPR.append(tp / (tp + fn))
                precision, recall, _ = precision_recall_curve(y_test == class_label, test_pred_proba[:, index])
                all_PR_CURVE.append(auc(recall, precision))
            data_dict['FPR'] = np.mean(all_FPR)
            data_dict['TPR'] = np.mean(all_TPR)
            data_dict['PR Curve'] = np.mean(all_PR_CURVE)

            df_results = df_results.append(data_dict, ignore_index=True)
    df_results.to_csv(results_dir + '/' + filename, index=False)
    df_results = df_results.iloc[0:0]


def test_models(random_state, external_split, internal_split, optimization_iterations):
    models = list(algorithms_dict.values())
    for filename in os.listdir(directory):
        run_test(filename, 'Results', models, random_state, external_split, internal_split, optimization_iterations)


def calc_plot_importance(X_no_dataset, Y_total):
    meta_classifier = train_on_all_datasets(X_no_dataset, Y_total)

    importance_types = ['weight', 'gain', 'cover']
    for type in importance_types:
        plot_importance(meta_classifier, importance_type=type, title='Feature importance: ' + type,
                        max_num_features=20)  # top 20 most important features
        plt.savefig('plots/' + type + '_importance.png')
        plt.clf()

    import shap

    # load JS visualization code to notebook
    shap.initjs()

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
    explainer = shap.TreeExplainer(meta_classifier)
    shap_values = explainer.shap_values(X_no_dataset)

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X_no_dataset.iloc[0, :])
    shap.summary_plot(shap_values, X_no_dataset, show=False)
    plt.savefig('plots/shap_values.png')
    plt.clf()


def test_meta_learner(results_directory):
    X_total, Y_total, X_no_dataset = read_meta_features(results_directory)
    accuracies = []
    for filename in os.listdir(results_directory):
        acc = acc4certain_dataset(filename.replace('.csv', ''), X_total, Y_total, X_no_dataset)
        accuracies.append(acc)
    accuracy = np.mean(accuracies)
    print('Meta Learner\'s Accuracy =', accuracy)

    calc_plot_importance(X_no_dataset, Y_total)


random_state = 42
external_split = 10
internal_split = 3
optimization_iterations = 50
test_models(random_state, external_split, internal_split, optimization_iterations)
test_meta_learner('Results')
