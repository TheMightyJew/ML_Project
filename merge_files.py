import glob
import pandas as pd
extension = 'csv'
all_filenames = [i for i in glob.glob('Results/*.{}'.format(extension))]
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
auc_combined_csv = combined_csv.loc[:, ['Dataset Name', 'Algorithm Name', 'Cross Validation[1-10]', 'AUC']]
write2dir = 'Summarized_Results'
combined_csv.to_csv(write2dir + '/combined_csv.csv', index=False, encoding='utf-8-sig')
auc_combined_csv.to_csv(write2dir + '/auc_combined_csv.csv', index=False, encoding='utf-8-sig')