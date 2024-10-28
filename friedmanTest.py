import csv
import numpy as np

def friedman_test(*args):
    """
    Perform the Friedman test on the provided data.
    
    Args:
        *args: multiple lists of related samples
    
    Returns:
        test_statistic: The test statistic for the Friedman test
        p_value: The p-value for the Friedman test
    """
    # Number of groups (k) and number of subjects (n)
    k = len(args)  # number of groups
    n = len(args[0])  # number of subjects (assuming all groups have the same number of subjects)
    
    # Step 1: Rank the data
    ranks = np.zeros((n, k))
    for i in range(n):
        ranks[i] = np.argsort(np.argsort([args[j][i] for j in range(k)])) + 1

    # Step 2: Calculate the sums of ranks for each group
    sum_ranks = np.sum(ranks, axis=0)
    
    # Step 3: Calculate the Friedman test statistic
    friedman_statistic = (12 / (n * k * (k + 1))) * np.sum(sum_ranks**2) - 3 * n * (k + 1)
    
    # Step 4: Calculate the degrees of freedom
    df = k - 1
    
    # Step 5: Calculate the p-value using the chi-squared distribution
    p_value = 1 - chi_square_cdf(friedman_statistic, df)
    
    return friedman_statistic, p_value

def chi_square_cdf(x, df):
    """
    Calculate the cumulative distribution function for the chi-squared distribution.
    This is a simple implementation of the CDF for a chi-squared distribution.
    """
    from scipy.special import gammainc, gamma
    
    return gammainc(df / 2, x / 2)  # Using scipy for CDF calculation

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

# Perform the Friedman test manually
friedman_auc_stat, p_auc = friedman_test(devnet_auc, iForest_auc)
friedman_precision_stat, p_precision = friedman_test(devnet_precision, iForest_precision)

# Print results
print(f"ROC Friedman Test: Statistic = {friedman_auc_stat}, p-value = {p_auc}")
print(f"Precision Friedman Test: Statistic = {friedman_precision_stat}, p-value = {p_precision}")
