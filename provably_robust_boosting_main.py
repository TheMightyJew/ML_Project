# Maksym Andriushchenko, Matthias Hein, Provably Robust Boosted Decision Stumps and Trees against Adversarial Attacks (NeurIPS 2019):
# https://paperswithcode.com/paper/provably-robust-boosted-decision-stumps-and-1

import numpy as np
import provably_robust_boosting.data as data
from provably_robust_boosting.tree_ensemble import TreeEnsemble

n_trees = 50  # total number of trees in the ensemble
model = 'robust_bound'  # robust tree ensemble
X_train, y_train, X_test, y_test, eps = data.all_datasets_dict['breast_cancer']()
X_train, X_test = data.convert_to_float32(X_train), data.convert_to_float32(X_test)

# initialize a tree ensemble with some hyperparameters
ensemble = TreeEnsemble(weak_learner='tree', n_trials_coord=X_train.shape[1],
                        lr=0.01, min_samples_split=10, min_samples_leaf=5, max_depth=4,
                        max_weight=1.0, idx_clsf=0)
# initialize gammas, per-example weights which are recalculated each iteration
gamma = np.ones(X_train.shape[0])
for i in range(1, n_trees + 1):
    # fit a new tree in order to minimize the robust loss of the whole ensemble
    weak_learner = ensemble.fit_tree(X_train, y_train, gamma, model, eps, depth=1)
    margin_prev = ensemble.certify_treewise(X_train, y_train, eps)  # needed for pruning
    ensemble.add_weak_learner(weak_learner)
    ensemble.prune_last_tree(X_train, y_train, margin_prev, eps, model)
    # calculate per-example weights for the next iteration
    gamma = np.exp(-ensemble.certify_treewise(X_train, y_train, eps))

    y_pred = ensemble.predict(X_test)
    # track generalization and robustness
    yf_test = y_test * ensemble.predict(X_test)
    min_yf_test = ensemble.certify_treewise(X_test, y_test, eps)
    if i == 1 or i % 5 == 0:
        print('Iteration: {}, test error: {:.2%}, upper bound on robust test error: {:.2%}'.format(
            i, np.mean(yf_test < 0.0), np.mean(min_yf_test < 0.0)))