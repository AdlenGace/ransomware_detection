# src/real_time.py

import time
import pandas as pd
from model import load_model
from feature_extraction import extract_features
from data_collection import collect_pcap_logs, preprocess_logs
from sklearn.preprocessing import StandardScaler
import joblib

# Load trained column names and scaler if available
try:
    trained_columns = joblib.load("data/models/trained_columns.pkl")
except FileNotFoundError:
    trained_columns = None
    print("[WARNING] Trained columns file not found. Feature alignment might fail.")

try:
    scaler = joblib.load("data/models/scaler.pkl")
except FileNotFoundError:
    scaler = None
    print("[WARNING] Scaler file not found. Scaling will be skipped.")

def monitor_system_logs(log_path, model_path):
    """
    Continuously monitor the system logs and make real-time predictions.
    Args:
        log_path (str): Path to the log directory
        model_path (str): Path to the trained model
    """
    print("[INFO] Starting real-time monitoring...")
    model = load_model(model_path)

    try:
        while True:
            print(f"[INFO] Processing {log_path}/capture.pcap...")
            logs = collect_pcap_logs(log_path)  # Use PCAP collection
            if logs.empty:
                print("[INFO] No new logs to process.")
                time.sleep(5)
                continue

            logs = preprocess_logs(logs)
            features = extract_features(logs)

            # Drop non-numeric identifiers if present
            if 'ip_address' in features.columns:
                features = features.drop(columns=['ip_address'])

            # One-hot encode categorical features like 'protocol'
            if 'protocol' in features.columns:
                features = pd.get_dummies(features, columns=['protocol'])

            # Align features with training columns
            if trained_columns:
                for col in trained_columns:
                    if col not in features.columns:
                        features[col] = 0  # Fill missing columns
                features = features[trained_columns]  # Reorder columns

            # Apply scaling if scaler was used during training
            if scaler:
                features = scaler.transform(features)

            # Predict activity
            predictions = model.predict(features)

            for i, prediction in enumerate(predictions):
                label = "malicious" if prediction == 1 or prediction == 'malicious' else "benign"
                status = "[ALERT]" if label == "malicious" else "[INFO]"
                print(f"{status} Entry {i} is {label}.")

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n[INFO] Real-time monitoring stopped by user.")

# Run the monitor
monitor_system_logs('data/raw_logs', 'data/models/random_forest_model.pkl')
