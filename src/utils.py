# utils.py

from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_features(features):
    """
    Scale features using StandardScaler.
    Args:
        features (pd.DataFrame): Features to scale
    Returns:
        pd.DataFrame: Scaled features
    """
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

def save_data(data, file_path):
    """
    Save processed data to a CSV file.
    Args:
        data (pd.DataFrame): Data to save
        file_path (str): Path where the data should be saved
    """
    data.to_csv(file_path, index=False)
