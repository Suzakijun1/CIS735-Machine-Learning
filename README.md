# CAN Bus Anomaly Detection

This repository contains scripts and data for detecting anomalies in CAN bus traffic using Isolation Forest. The project compares normal driving conditions (FreeDrivingData) with potential attack scenarios (FuzzyDataset).

## Features

- Preprocessing and feature extraction from CAN bus data.
- Anomaly detection using Isolation Forest.
- Comparison of normal and abnormal CAN traffic patterns.
- Visualization of anomaly scores and key payload features.

## Files

- **`visualization.py`**: Python script for data visualization, anomaly detection, and comparison between datasets.
- **`enhanced_free_driving_data.csv`**: Processed CAN bus data representing normal driving conditions.
- **`enhanced_fuzzy_data.csv`**: Processed CAN bus data representing an attack scenario (fuzzy dataset).
- **`FreeDrivingData.txt`**: Raw CAN data for normal driving.
- **`FuzzyDataset.txt`**: Raw CAN data for a fuzzy attack scenario.

## Requirements

- Python 3.8+
- Required libraries:
  - `pandas`
  - `matplotlib`
  - `scikit-learn`

Install dependencies with:

```bash
pip install pandas matplotlib scikit-learn
```
