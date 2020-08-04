# Maksym Andriushchenko, Matthias Hein, Provably Robust Boosted Decision Stumps and Trees against Adversarial Attacks (NeurIPS 2019):
# https://paperswithcode.com/paper/provably-robust-boosted-decision-stumps-and-1
import inspect
import copy
import numpy as np
from xbart import XBART

class Wrapper_xbart:

    def __init__(self, num_trees=100, num_sweeps=40, burnin=15):
        self.num_trees = num_trees
        self.num_sweeps = num_sweeps
        self.burnin = burnin
        self.model = None

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        params = copy.copy(self.__dict__)
        del params['model']
        return params

    '''
    def predict_proba(self, X_test):
        #usable_X_test = X_test.to_numpy()
        y_pred = self.model.(X_test)
        y_pred = (y_pred + 1) * 0.5
        return np.array([[1-x, x] for x in y_pred])
    '''
    def predict(self, X_test):
        #return np.round(self.predict_proba(X_test)[:, 1])
        return np.round(self.model.predict(X_test))
        '''
        # track generalization and robustness
        yf_test = y_test * self.model.predict(X_test)
        min_yf_test = ensemble.certify_treewise(X_test, y_test, self.eps)
        if i == 1 or i % 5 == 0:
            print('Iteration: {}, test error: {:.2%}, upper bound on robust test error: {:.2%}'.format(
                i, np.mean(yf_test < 0.0), np.mean(min_yf_test < 0.0)))
        '''

    def fit(self, X_train, y_train):
        #usable_X_train = X_train.to_numpy()
        #y_train[y_train == 0] = -1
        ensemble = XBART(num_trees=self.num_trees, num_sweeps=self.num_sweeps, burnin=self.burnin)
        ensemble.fit(X_train, y_train)
        self.model = ensemble
