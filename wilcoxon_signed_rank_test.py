import csv
import scipy.stats as stats
import numpy as np

iForest_result = 'results/auc_performance_cl0.5.csv'
devnet_result = 'results/result_kfold.csv'

# Read iForest results
with open(iForest_result, 'r') as f:
    reader = csv.reader(f)
    iForest_data = list(reader)
    iForest_auc = []
    iForest_precision = []
    for row in iForest_data:
        iForest_auc.append(float(row[7]))
        iForest_precision.append(float(row[9]))

# Read DevNet results
with open(devnet_result, 'r') as f:
    reader = csv.reader(f)
    devnet_data = list(reader)
    devnet_auc = []
    devnet_precision = []
    for row in devnet_data:
        devnet_auc.append(float(row[7]))
        devnet_precision.append(float(row[9]))

# Function to calculate z-value from Wilcoxon signed-rank test
def wilcoxon_signed_rank_test(data1, data2):
    # Calculate differences and absolute differences
    differences = np.array(data1) - np.array(data2)
    abs_differences = np.abs(differences)
    
    # Rank the absolute differences (ignoring zeros)
    ranks = stats.rankdata(abs_differences)
    
    # Assign signs to ranks
    signed_ranks = ranks * np.sign(differences)
    
    # Calculate W+ (sum of positive ranks) and W- (sum of negative ranks)
    W_pos = np.sum(signed_ranks[signed_ranks > 0])
    W_neg = np.sum(np.abs(signed_ranks[signed_ranks < 0]))
    
    # Use the smaller of W+ and W-
    W = min(W_pos, W_neg)
    
    # Sample size
    n = len(differences) - np.sum(differences == 0)  # exclude zero differences
    
    # Mean and standard deviation of W under the null hypothesis
    mean_W = n * (n + 1) / 4
    std_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    
    # Calculate z-value (normal approximation)
    z_value = (W - mean_W) / std_W
    
    # Two-tailed p-value for the z-value
    p_value = 2 * stats.norm.sf(np.abs(z_value))
    
    return z_value, p_value, W_pos, W_neg

# Perform the Wilcoxon signed-rank test and calculate z-values for AUC
z_auc, p_auc, W_pos_auc, W_neg_auc = wilcoxon_signed_rank_test(devnet_auc, iForest_auc)
print(f"AUC Wilcoxon Test: z-value = {z_auc}, p-value = {p_auc}, W+ = {W_pos_auc}, W- = {W_neg_auc}")

# Perform the Wilcoxon signed-rank test and calculate z-values for Precision
z_precision, p_precision, W_pos_precision, W_neg_precision = wilcoxon_signed_rank_test(devnet_precision, iForest_precision)
print(f"Precision Wilcoxon Test: z-value = {z_precision}, p-value = {p_precision}, W+ = {W_pos_precision}, W- = {W_neg_precision}")
