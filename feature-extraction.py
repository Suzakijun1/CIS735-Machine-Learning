import pandas as pd
import numpy as np

# Specify the file path
file_path = r'C:\Users\Dylan\Desktop\survival\dataset\Soul\Flooding_dataset_KIA.txt'

# Load the file with all columns and correct column names
can_data = pd.read_csv(file_path, delimiter=',', header=None,
                       names=['timestamp', 'can_id', 'dlc', 'data_bytes1', 'data_bytes2', 'data_bytes3', 
                              'data_bytes4', 'data_bytes5', 'data_bytes6', 'data_bytes7', 'data_bytes8', 'direction'])

# Convert timestamp to float for numerical operations
can_data['timestamp'] = can_data['timestamp'].astype(float)

# Define a function to clean non-hexadecimal values from data bytes columns
def clean_data_byte(byte):
    try:
        # Ensure byte is a string before attempting conversion
        return int(str(byte), 16)  # Convert to integer if valid hex
    except ValueError:
        return np.nan  # Return NaN for invalid hex values (like 'R')

# Apply cleaning to each data byte column
for col in ['data_bytes1', 'data_bytes2', 'data_bytes3', 'data_bytes4', 
            'data_bytes5', 'data_bytes6', 'data_bytes7', 'data_bytes8']:
    can_data[col] = can_data[col].apply(clean_data_byte)

# Drop rows with NaN in any data byte column to remove rows with 'R' or other invalid values
can_data.dropna(subset=['data_bytes1', 'data_bytes2', 'data_bytes3', 'data_bytes4', 
                        'data_bytes5', 'data_bytes6', 'data_bytes7', 'data_bytes8'], inplace=True)

# Combine data bytes columns into a single list for each row
can_data['data_bytes'] = can_data[['data_bytes1', 'data_bytes2', 'data_bytes3', 'data_bytes4', 
                                   'data_bytes5', 'data_bytes6', 'data_bytes7', 'data_bytes8']].values.tolist()

# Drop the individual data byte columns as they are now combined into 'data_bytes'
can_data.drop(columns=['data_bytes1', 'data_bytes2', 'data_bytes3', 'data_bytes4', 
                       'data_bytes5', 'data_bytes6', 'data_bytes7', 'data_bytes8', 'direction'], inplace=True)

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
can_data.to_csv('enhanced_can_data.csv', index=False)




