# Maksym Andriushchenko, Matthias Hein, Provably Robust Boosted Decision Stumps and Trees against Adversarial Attacks (NeurIPS 2019):
# https://paperswithcode.com/paper/provably-robust-boosted-decision-stumps-and-1
import inspect
import copy
import numpy as np
from provably_robust_boosting.tree_ensemble import TreeEnsemble

class Wrapper_provably_robust_boosting:

    def __init__(self, lr=0.01, min_samples_split=10, min_samples_leaf=5, max_depth=4, max_weight=1.0, idx_clsf=0, n_trees=50):
        self.eps = 0.35
        self.lr = lr
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_weight = max_weight
        self.idx_clsf = idx_clsf
        self.n_trees = n_trees
        self.model = None

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        params = copy.copy(self.__dict__)
        del params['eps']
        del params['model']
        return params

    def predict_proba(self, X_test):
        usable_X_test = X_test.to_numpy()
        y_pred = self.model.predict(usable_X_test)
        y_pred = (y_pred + 1) * 0.5
        return np.array([[1-x, x] for x in y_pred])

    def predict(self, X_test):
        return np.round(self.predict_proba(X_test)[:, 1])
        '''
        # track generalization and robustness
        yf_test = y_test * self.model.predict(X_test)
        min_yf_test = ensemble.certify_treewise(X_test, y_test, self.eps)
        if i == 1 or i % 5 == 0:
            print('Iteration: {}, test error: {:.2%}, upper bound on robust test error: {:.2%}'.format(
                i, np.mean(yf_test < 0.0), np.mean(min_yf_test < 0.0)))
        '''

    def fit(self, X_train, y_train):
        model_name = 'robust_bound'  # robust tree ensemble

        usable_X_train = X_train.to_numpy()
        y_train[y_train == 0] = -1
        # initialize a tree ensemble with some hyperparameters
        ensemble = TreeEnsemble(weak_learner='tree', n_trials_coord=usable_X_train.shape[1],
                                lr=self.lr, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth,
                                max_weight=self.max_weight, idx_clsf=self.idx_clsf)
        # initialize gammas, per-example weights which are recalculated each iteration
        gamma = np.ones(usable_X_train.shape[0])
        for i in range(1, self.n_trees + 1):
            # fit a new tree in order to minimize the robust loss of the whole ensemble
            weak_learner = ensemble.fit_tree(usable_X_train, y_train, gamma, model_name, self.eps, depth=1)
            margin_prev = ensemble.certify_treewise(usable_X_train, y_train, self.eps)  # needed for pruning
            ensemble.add_weak_learner(weak_learner)
            ensemble.prune_last_tree(usable_X_train, y_train, margin_prev, self.eps, model_name)
            # calculate per-example weights for the next iteration
            gamma = np.exp(-ensemble.certify_treewise(usable_X_train, y_train, self.eps))
            # finished the iteration

        #save the model for farther usage
        self.model = ensemble
