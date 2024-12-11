import pandas as pd
import numpy as np

# Specify the file path
file_path = r'C:\Users\Dylan\Desktop\survival\dataset\Soul\FreeDrivingData_20180112_KIA.txt'

# Load the file with correct column names
can_data = pd.read_csv(file_path, delimiter=',', header=None,
                       names=['timestamp', 'can_id', 'dlc', 'payload'])

# Convert timestamp to float for numerical operations
can_data['timestamp'] = can_data['timestamp'].astype(float)

# Parse payload into a list of bytes
def parse_payload(payload, dlc):
    try:
        # Split the payload string into individual byte values
        payload_list = payload.split(' ')
        # Ensure the payload length matches the DLC; pad with zeros if necessary
        return [int(byte, 16) for byte in payload_list[:int(dlc)]] + [0] * (int(dlc) - len(payload_list))
    except Exception as e:
        return [0] * int(dlc)  # Handle cases where parsing fails

# Apply payload parsing
can_data['data_bytes'] = can_data.apply(lambda row: parse_payload(row['payload'], row['dlc']), axis=1)

# Drop the original payload column as it is now parsed
can_data.drop(columns=['payload'], inplace=True)

# Sort by timestamp for accurate interval calculations
can_data.sort_values(by='timestamp', inplace=True)

# Calculate the time interval between consecutive messages for the same CAN ID
can_data['time_diff'] = can_data.groupby('can_id')['timestamp'].diff().fillna(0)

# Function to calculate mean, variance, and unique count of payload data
def calculate_payload_stats(data_bytes):
    return pd.Series({
        'payload_mean': np.mean(data_bytes),
        'payload_variance': np.var(data_bytes),
        'payload_unique': len(set(data_bytes))
    })

# Apply the function to the data_bytes column to extract payload statistics
payload_stats = can_data['data_bytes'].apply(calculate_payload_stats)
can_data = pd.concat([can_data, payload_stats], axis=1)

# Display the DataFrame with new features
print("CAN Data with Enhanced Features:\n", can_data.head())

# Save the processed data with new features to a .csv file if needed
can_data.to_csv('enhanced_free_driving_data.csv', index=False)
