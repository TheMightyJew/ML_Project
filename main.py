import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from WrappedModels.wrapper_provably_robust_boosting import Wrapper_provably_robust_boosting as wprb
from scipy.stats import uniform
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.multiclass import OneVsRestClassifier

directory = 'classification_datasets'
models = [(wprb, OneVsRestClassifier(wprb()), dict(estimator__min_samples_split=uniform(loc=10, scale=90), estimator__max_depth=uniform(loc=2, scale=3)))]
random_state = 42
external_split = 10
internal_split = 3
optimization_iterations = 50

for filename in os.listdir(directory):
    df = pd.read_csv(directory + '/' + filename)
    #df.columns = list(df.columns[:-1]) + ['Class']
    df.columns = [x for x in range(len(df.columns[:-1]))] + ['Class']
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1:]
    kf = KFold(n_splits=external_split, random_state=random_state)
    print(filename)
    for train_index, test_index in kf.split(X):
        x_train = X.iloc[train_index]
        y_train = Y.iloc[train_index]
        x_test = X.iloc[test_index]
        y_test = Y.iloc[test_index]
        for model_class, model, model_dict in models:
            # distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
            distributions = model_dict
            randomSearcher = RandomizedSearchCV(model, distributions, random_state=random_state, cv=internal_split,
                                                n_iter=optimization_iterations, scoring=make_scorer(accuracy_score))
            randomSearcher.fit(x_train, y_train)
            best_model = OneVsRestClassifier(model_class(**randomSearcher.best_params_))
