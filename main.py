import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from WrappedModels.wrapper_provably_robust_boosting import Wrapper_provably_robust_boosting as wprb
from scipy.stats import uniform, randint, rv_discrete
from random import randrange
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.multiclass import OneVsRestClassifier
import lightgbm
from xbart import XBART
from WrappedModels.wrapper_xbart import Wrapper_xbart as wxbart

directory = 'classification_datasets'
models = []
'''
num_leaves_k = [5, 10, 50, 100]
num_leaves_dist = rv_discrete(name='num_leaves_dist', values=(num_leaves_k, [1/len(num_leaves_k)] * len(num_leaves_k)))
n_estimators_k = [20, 50, 100, 200]
n_estimators_dist = rv_discrete(name='n_estimators_dist', values=(n_estimators_k, [1/len(n_estimators_k)] * len(n_estimators_k)))
dist_dict = dict(num_leaves=num_leaves_dist, n_estimators=n_estimators_dist)
'''
'''
xk = [5, 10, 50, 100, 200, 500, 1000]
pk = [1/len(xk)] * len(xk)
min_samples_split_dist = rv_discrete(name='min_samples_split_dist', values=(xk, pk))
dist_dict = dict(estimator__min_samples_split=min_samples_split_dist, estimator__max_depth=randint(3, 6))
'''

num_trees_k = [20, 50, 100, 200]
num_trees_dist = rv_discrete(name='num_trees_dist', values=(num_trees_k, [1/len(num_trees_k)] * len(num_trees_k)))
num_sweeps_k = [20, 40, 60, 80]
num_sweeps_dist = rv_discrete(name='num_sweeps_dist', values=(num_sweeps_k, [1/len(num_sweeps_k)] * len(num_sweeps_k)))
dist_dict = dict(num_trees=num_trees_k, num_sweeps=num_sweeps_dist)
#num_trees=100, num_sweeps=40, burnin=15
models.append((wxbart, wxbart(), dist_dict))
#models.append((lightgbm.LGBMClassifier, lightgbm.LGBMClassifier(), dist_dict))
#models.append((wprb, OneVsRestClassifier(wprb()), dist_dict))

random_state = 42
external_split = 10
internal_split = 3
optimization_iterations = 5

for filename in os.listdir(directory):
    #filename = 'iris.csv'
    df = pd.read_csv(directory + '/' + filename)
    #df.columns = list(df.columns[:-1]) + ['Class']
    df.columns = [x for x in range(len(df.columns[:-1]))] + ['Class']
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1:]
    kf = KFold(n_splits=external_split, random_state=random_state)
    print(filename)
    for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
        print("fold index =", fold_index)
        x_train = X.iloc[train_index]
        y_train = Y.iloc[train_index]
        x_test = X.iloc[test_index]
        y_test = Y.iloc[test_index]
        for model_class, model, model_dict in models:
            # distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
            distributions = model_dict
            randomSearcher = RandomizedSearchCV(model, distributions, random_state=random_state, cv=internal_split,
                                                n_iter=optimization_iterations, scoring=make_scorer(accuracy_score))
            randomSearcher.fit(x_train, y_train.values.ravel())

            if model_class is wprb:
                params = {k.replace("estimator__", ""): v for k, v in randomSearcher.best_params_.items()}
                best_model = OneVsRestClassifier(model_class(**params))
            else:
                params = randomSearcher.best_params_
                best_model = model_class(**params)
            best_model.fit(x_train, y_train.values.ravel())
            print("best params:", params)
            print("train accuracy:", accuracy_score(y_train, best_model.predict(x_train)))
            print("test accuracy:", accuracy_score(y_test, best_model.predict(x_test)))
            print()
