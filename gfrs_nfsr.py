import numpy as np

# Equation 7: Distance from pattern to class mean
def distance_to_mean(x_ki, mean_k, var_k):
    return (x_ki - mean_k) ** 2 / var_k

# Equation 8: Membership value calculation
def membership_value(D_ki, w1):
    return 1 / (1 + np.exp(D_ki * w1))

# Equation 9: Influence value of class k
def influence_value(x_ki_list, mean_k, w1, membership_values):
    numerator = 0
    denominator = 0
    
    for i in range(len(x_ki_list)):
        x_ki = x_ki_list[i]
        mki = membership_values[i]    
        numerator += (x_ki - mean_k) ** 2 * np.exp(mki * w1)
        denominator += np.exp(mki * w1)
    
    return numerator / denominator if denominator != 0 else 0

# Equation 10: Similarity value between two patterns for a given feature
def similarity_value(x_ki, y, mean_k, a_j):
    return (a_j(x_ki) - a_j(y) + a_j(mean_k)) / a_j(mean_k)


x_ki_list = [1.2, 2.4, 1.9]
mean_k = 2.0
var_k = 0.5
w1 = 1.5

distances = [distance_to_mean(x_ki, mean_k, var_k) for x_ki in x_ki_list]

membership_values = [membership_value(D_ki, w1) for D_ki in distances]

influence_k = influence_value(x_ki_list, mean_k, w1, membership_values)

def a_j(x): return x  # identity function, just for example
y = 2.1  # Another pattern
similarities = [similarity_value(x_ki, y, mean_k, a_j) for x_ki in x_ki_list]

print("Distances:", distances)
print("Membership values:", membership_values)
print("Influence value of class k:", influence_k)
print("Similarities:", similarities)