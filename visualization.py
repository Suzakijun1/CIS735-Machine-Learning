# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import IsolationForest

# # Load the enhanced data with new features
# file_path = r'C:\Users\Dylan\Documents\Syracuse\cis735-final\enhanced_can_data.csv'  # Make sure this file path is correct
# can_data = pd.read_csv(file_path)

# # Plot histograms for payload features
# plt.figure(figsize=(10, 5))
# plt.hist(can_data['payload_mean'], bins=30, alpha=0.7, label='Payload Mean')
# plt.hist(can_data['payload_variance'], bins=30, alpha=0.7, label='Payload Variance')
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.legend()
# plt.title("Histogram of Payload Mean and Variance")
# plt.show()

# # Apply Isolation Forest for anomaly detection
# features = can_data[['time_diff', 'payload_mean', 'payload_variance', 'payload_unique']]

# # Train isolation forest
# iso_forest = IsolationForest(contamination=0.1, random_state=42)
# can_data['anomaly_score'] = iso_forest.fit_predict(features)

# # Visualize anomalies in the dataset
# anomalies = can_data[can_data['anomaly_score'] == -1]
# plt.scatter(can_data.index, can_data['payload_mean'], label='Normal')
# plt.scatter(anomalies.index, anomalies['payload_mean'], color='r', label='Anomaly')
# plt.xlabel("Index")
# plt.ylabel("Payload Mean")
# plt.title("Anomaly Detection in Payload Mean")
# plt.legend()
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# Load the enhanced data for both datasets
free_driving_path = r'C:\Users\Dylan\Documents\Syracuse\cis735-final\enhanced_free_driving_data.csv'
fuzzy_path = r'C:\Users\Dylan\Documents\Syracuse\cis735-final\enhanced_can_data.csv'

free_driving_data = pd.read_csv(free_driving_path)
fuzzy_data = pd.read_csv(fuzzy_path)

# Select features for normalization
features = ['time_diff', 'payload_mean', 'payload_variance', 'payload_unique']

# Normalize both datasets
scaler = MinMaxScaler()
free_driving_features = scaler.fit_transform(free_driving_data[features])
fuzzy_features = scaler.transform(fuzzy_data[features])

# Train Isolation Forest on FreeDrivingData
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(free_driving_features)

# Score both datasets
free_driving_data['anomaly_score'] = iso_forest.decision_function(free_driving_features)
free_driving_data['is_anomaly'] = iso_forest.predict(free_driving_features)  # -1 for anomalies

fuzzy_data['anomaly_score'] = iso_forest.decision_function(fuzzy_features)
fuzzy_data['is_anomaly'] = iso_forest.predict(fuzzy_features)  # -1 for anomalies

# Compare Anomaly Rates
free_driving_anomaly_rate = (free_driving_data['is_anomaly'] == -1).mean()
fuzzy_anomaly_rate = (fuzzy_data['is_anomaly'] == -1).mean()

print(f"Free Driving Anomaly Rate: {free_driving_anomaly_rate:.2%}")
print(f"Fuzzy Dataset Anomaly Rate: {fuzzy_anomaly_rate:.2%}")

# Visualize Anomaly Score Distributions
plt.figure(figsize=(12, 6))
plt.hist(free_driving_data['anomaly_score'], bins=50, alpha=0.7, label='Free Driving Data')
plt.hist(fuzzy_data['anomaly_score'], bins=50, alpha=0.7, label='Flooding Dataset')
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.legend()
plt.title("Comparison of Anomaly Scores")
plt.show()

# Visualize Anomalies in Payload Mean for Both Datasets
plt.figure(figsize=(12, 6))
plt.scatter(free_driving_data.index, free_driving_data['payload_mean'], label='Free Driving - Normal', alpha=0.7)
plt.scatter(free_driving_data[free_driving_data['is_anomaly'] == -1].index, 
            free_driving_data[free_driving_data['is_anomaly'] == -1]['payload_mean'], 
            color='r', label='Free Driving - Anomaly', alpha=0.7)

plt.scatter(fuzzy_data.index, fuzzy_data['payload_mean'], label='Flooding - Normal', alpha=0.7)
plt.scatter(fuzzy_data[fuzzy_data['is_anomaly'] == -1].index, 
            fuzzy_data[fuzzy_data['is_anomaly'] == -1]['payload_mean'], 
            color='purple', label='Flooding - Anomaly', alpha=0.7)

plt.xlabel("Index")
plt.ylabel("Payload Mean")
plt.title("Anomaly Detection in Payload Mean (Free Driving vs. Flooding)")
plt.legend()
plt.show()
