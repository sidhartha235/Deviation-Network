import csv
import scipy.stats as stats

iForest_result = 'results/auc_performance_cl0.5.csv'
devnet_result = 'results/result_kfold.csv'

with open(iForest_result, 'r') as f:
    reader = csv.reader(f)
    iForest_data = list(reader)
    iForest_auc = []
    iForest_precision = []
    for row in iForest_data:
        iForest_auc.append(float(row[7]))
        iForest_precision.append(float(row[9]))

# print("IForest AUC") 
# print(iForest_auc)
# print("IForest Precision") 
# print(iForest_precision)

with open(devnet_result, 'r') as f:
    reader = csv.reader(f)
    devnet_data = list(reader)
    devnet_auc = []
    devnet_precision = []
    for row in devnet_data:
        devnet_auc.append(float(row[7]))
        devnet_precision.append(float(row[9]))

# print("Devnet AUC") 
# print(devnet_auc)
# print("Devnet Precision")
# print(devnet_precision)

# Perform the t-test
t_auc, p_auc = stats.ttest_rel(devnet_auc, iForest_auc)
t_precision, p_precision = stats.ttest_rel(devnet_precision, iForest_precision)
print(f"AUC T-Test : {t_auc} and p-value : {p_auc}")
print(f"Precision T-Test : {t_precision} and p-value : {p_precision}")