from sklearn.ensemble import IsolationForest
import time
from utils import dataLoading, aucPerformance, writeResults, get_data_from_svmlight_file
import os

def writeResults(name, n_samples, dim, n_samples_trn, n_outliers_trn, n_outliers, depth, rauc, ap, std_auc, std_ap, train_time, test_time, path = "./results/auc_performance_cl0.5.csv"):    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Write to the file
    with open(path, 'a') as csv_file:
        row = name + "," + str(n_samples)+ ","  + str(dim) + ',' + str(n_samples_trn) + ','+ str(n_outliers_trn) + ','+ str(n_outliers)  + ',' + str(depth)+ "," + str(rauc) +"," + str(std_auc) + "," + str(ap) +"," + str(std_ap)+"," + str(train_time)+"," + str(test_time) + "\n"
        csv_file.write(row)

    # Print the results
    print(f"Dataset: {name}")
    print(f"Number of samples: {n_samples}")
    print(f"Dimensions: {dim}")
    print(f"Training samples: {n_samples_trn}")
    print(f"Outliers in training: {n_outliers_trn}")
    print(f"Outliers: {n_outliers}")
    print(f"Depth: {depth}")
    print(f"ROC AUC: {rauc} (± {std_auc})")
    print(f"Average Precision: {ap} (± {std_ap})")
    print(f"Training time: {train_time}s")
    print(f"Testing time: {test_time}s")

# Import utility functions (assuming you have utils.py available)
# from utils import dataLoading, aucPerformance, writeResults

# Load your dataset (modify the path to your dataset)
X, labels = dataLoading(".//Deviation-Network//dataset//annthyroid_21feat_normalised.csv")

# Initialize the Isolation Forest
iso_forest = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', random_state=42)

# Train the Isolation Forest
start_train_time = time.time()
iso_forest.fit(X)
train_time = time.time() - start_train_time

# Get anomaly scores (lower means more anomalous)
start_test_time = time.time()
anomaly_scores = -iso_forest.decision_function(X)  # Flip the sign so that higher values indicate anomalies
test_time = time.time() - start_test_time

# Evaluate performance using ROC-AUC and Precision-Recall metrics
roc_auc, ap = aucPerformance(anomaly_scores, labels)

# Optionally, log the results
writeResults(
    name="IsolationForest", 
    n_samples=X.shape[0], 
    dim=X.shape[1], 
    n_samples_trn=len(X), 
    n_outliers_trn=(labels == 1).sum(),  # Assuming 1 represents outliers
    n_outliers=(labels == 1).sum(),
    depth=iso_forest.get_params()['n_estimators'], 
    rauc=roc_auc, 
    ap=ap, 
    std_auc=0,  # Placeholder (can compute std if needed)
    std_ap=0,  # Placeholder (can compute std if needed)
    train_time=train_time, 
    test_time=test_time
)

# Print the results to the console (already handled by writeResults)
