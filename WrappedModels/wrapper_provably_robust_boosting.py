# Maksym Andriushchenko, Matthias Hein, Provably Robust Boosted Decision Stumps and Trees against Adversarial Attacks (NeurIPS 2019):
# https://paperswithcode.com/paper/provably-robust-boosted-decision-stumps-and-1

import numpy as np
from provably_robust_boosting.tree_ensemble import TreeEnsemble


class Wrapper_provably_robust_boosting:

    def __init__(self, lr=0.01, min_samples_split=10, min_samples_leaf=5, max_depth=4, max_weight=1.0, idx_clsf=0, n_trees=50):
        self.eps = 0
        self.lr = lr
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_weight = max_weight
        self.idx_clsf = idx_clsf
        self.n_trees = n_trees
        self.model = None

    def set_params(self, lr=0.01, min_samples_split=10, min_samples_leaf=5, max_depth=4, max_weight=1.0, idx_clsf=0, n_trees=50):
        self.eps = 0
        self.lr = lr
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_weight = max_weight
        self.idx_clsf = idx_clsf
        self.n_trees = n_trees
        self.model = None

    def predict(self, X_test):
        return self.model.predict(X_test)
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

        # initialize a tree ensemble with some hyperparameters
        ensemble = TreeEnsemble(weak_learner='tree', n_trials_coord=X_train.shape[1],
                                lr=self.lr, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth,
                                max_weight=self.max_weight, idx_clsf=self.idx_clsf)
        # initialize gammas, per-example weights which are recalculated each iteration
        gamma = np.ones(X_train.shape[0])
        for i in range(1, self.n_trees + 1):
            # fit a new tree in order to minimize the robust loss of the whole ensemble
            weak_learner = ensemble.fit_tree(X_train, y_train, gamma, model_name, self.eps, depth=1)
            margin_prev = ensemble.certify_treewise(X_train, y_train, self.eps)  # needed for pruning
            ensemble.add_weak_learner(weak_learner)
            ensemble.prune_last_tree(X_train, y_train, margin_prev, self.eps, model_name)
            # calculate per-example weights for the next iteration
            gamma = np.exp(-ensemble.certify_treewise(X_train, y_train, self.eps))
            # finished the iteration

        #save the model for farther usage
        self.model = ensemble