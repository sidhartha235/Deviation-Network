import csv
import math

def calculate_ttest(devnet_data, iForest_data):
    n = len(devnet_data)
    
    # Step 1: Calculate differences
    differences = [devnet_data[i] - iForest_data[i] for i in range(n)]
    
    # Step 2: Calculate mean of differences
    mean_diff = sum(differences) / n
    
    # Step 3: Calculate standard deviation of differences
    variance_diff = sum((d - mean_diff) ** 2 for d in differences) / (n - 1)  # sample variance
    std_dev_diff = math.sqrt(variance_diff)

    # Step 4: Calculate t-statistic
    t_statistic = mean_diff / (std_dev_diff / math.sqrt(n))
    
    # Step 5: Degrees of freedom
    degrees_of_freedom = n - 1
    
    return t_statistic, degrees_of_freedom

# File paths
iForest_result = 'results/iForest_thyroid.csv'
devnet_result = 'results/result_devnet_kfold.csv'

# Reading iForest data
with open(iForest_result, 'r') as f:
    reader = csv.reader(f)
    iForest_data = list(reader)
    iForest_auc = []
    iForest_precision = []
    for row in iForest_data:
        iForest_auc.append(float(row[7]))
        iForest_precision.append(float(row[9]))

# Reading devnet data
with open(devnet_result, 'r') as f:
    reader = csv.reader(f)
    devnet_data = list(reader)
    devnet_auc = []
    devnet_precision = []
    for row in devnet_data:
        devnet_auc.append(float(row[7]))
        devnet_precision.append(float(row[9]))

# Ensure data lengths match
if len(devnet_auc) != len(iForest_auc):
    raise ValueError("AUC data length mismatch: devnet_auc and iForest_auc must have the same length.")
if len(devnet_precision) != len(iForest_precision):
    raise ValueError("Precision data length mismatch: devnet_precision and iForest_precision must have the same length.")

# Perform the t-tests manually
t_auc, df_auc = calculate_ttest(devnet_auc, iForest_auc)
t_precision, df_precision = calculate_ttest(devnet_precision, iForest_precision)

# Print results
print(f"AUC T-Test: t-statistic = {t_auc}, degrees of freedom = {df_auc}")
print(f"Precision T-Test: t-statistic = {t_precision}, degrees of freedom = {df_precision}")
