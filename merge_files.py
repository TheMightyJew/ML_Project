import glob
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_wilcoxon


def merge_results():
    extension = 'csv'
    all_filenames = [i for i in glob.glob('Results/*.{}'.format(extension))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    auc_combined_csv = combined_csv.loc[:, ['Dataset Name', 'Algorithm Name', 'Cross Validation[1-10]', 'AUC']]
    write2dir = 'Summarized_Results'
    combined_csv.to_csv(write2dir + '/combined_csv.csv', index=False, encoding='utf-8-sig')
    auc_combined_csv.to_csv(write2dir + '/auc_combined_csv.csv', index=False, encoding='utf-8-sig')


def auc_for_algo(df, average=True):
    algorithms_auc = {}
    for algorithm in df['Algorithm Name'].unique():
        if average:
            aucs = []
            for database in df['Dataset Name'].unique():
                aucs.append(df[df['Dataset Name'] == database][df['Algorithm Name'] == algorithm]['AUC'].mean())
            algorithms_auc[algorithm] = aucs
        else:
            algorithms_auc[algorithm] = list(df[df['Algorithm Name'] == algorithm]['AUC'])
    return algorithms_auc


def print_friedman_Test(algorithms_auc):
    stat, p = friedmanchisquare(*algorithms_auc.values())
    print('stat=', round(stat, 3), '| p=', p)
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')


def print_descriptive_statistics(algorithms_auc):
    for index, algorithm in enumerate(algorithms_auc.keys()):
        print(index + 1, algorithm, ':\t Mean=', round(np.mean(algorithms_auc[algorithm]), 3), '| STD=',
              round(np.std(algorithms_auc[algorithm]), 3))


def print_posthoc(algorithms_auc):
    print(posthoc_wilcoxon(list(algorithms_auc.values())).round(3))


df = pd.read_csv('Summarized_Results/auc_combined_csv.csv')
algorithms_auc = auc_for_algo(df)
print_friedman_Test(algorithms_auc)
print_descriptive_statistics(algorithms_auc)
print_posthoc(algorithms_auc)