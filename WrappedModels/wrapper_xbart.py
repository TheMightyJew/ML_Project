# Maksym Andriushchenko, Matthias Hein, Provably Robust Boosted Decision Stumps and Trees against Adversarial Attacks (NeurIPS 2019):
# https://paperswithcode.com/paper/provably-robust-boosted-decision-stumps-and-1
import inspect
import copy
import numpy as np
from xbart import XBART
import sys


class Wrapper_xbart:

    def __init__(self, num_trees=100, num_sweeps=40, burnin=15):
        self.num_trees = num_trees
        self.num_sweeps = num_sweeps
        self.burnin = burnin
        self.model = None
        self.labels = None

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        params = copy.copy(self.__dict__)
        del params['model']
        del params['labels']
        return params

    def predict_proba(self, X_test):
        predictions = self.model.predict(X_test)
        all_probas = []
        for prediction in predictions:
            if prediction in self.labels:
                final_proba = [0 if x != prediction else 1 for x in self.labels]
            else:
                deltas = []
                for label in self.labels:
                    deltas.append(abs(label - prediction))
                probas = [delta / sum(deltas) for delta in deltas]
                reverse_probas = [1 / proba for proba in probas]
                final_proba = [reverse_proba / sum(reverse_probas) for reverse_proba in reverse_probas]
            all_probas.append(final_proba)
        return np.array(all_probas)

    def predict(self, X_test):
        return np.round(self.model.predict(X_test))

    def fit(self, X_train, y_train):
        self.labels = np.unique(y_train)
        ensemble = XBART(num_trees=self.num_trees, num_sweeps=self.num_sweeps, burnin=self.burnin)
        ensemble.fit(X_train, y_train)
        self.model = ensemble
