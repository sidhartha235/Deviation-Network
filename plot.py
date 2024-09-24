import pandas as pd
import matplotlib.pyplot as plt

# Define column names based on your data structure
columns = ['name', 'n_samples', 'dim', 'n_samples_trn', 'n_outliers_trn', 'n_outliers', 'depth', 
           'rauc', 'std_auc', 'ap', 'std_ap', 'train_time', 'test_time']

# Load the CSV file
file_path = 'results/result.csv'
df = pd.read_csv(file_path, names=columns)

# Plot ROC AUC and Average Precision
plt.figure(figsize=(10, 6))

# Plot ROC AUC
plt.plot(df['rauc'], label='ROC AUC', marker='o')

# Plot Average Precision
plt.plot(df['ap'], label='Average Precision', marker='s')

# Labels and Title
plt.xlabel('Index')
plt.ylabel('Score')
plt.title('ROC AUC and Average Precision Over Time')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Plot Training and Testing Time
plt.figure(figsize=(10, 6))

# Plot Training Time
plt.plot(df['train_time'], label='Training Time', marker='o')

# Plot Testing Time
plt.plot(df['test_time'], label='Testing Time', marker='s')

# Labels and Title
plt.xlabel('Index')
plt.ylabel('Time (s)')
plt.title('Training and Testing Time Over Time')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
