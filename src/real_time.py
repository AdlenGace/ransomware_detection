# real_time.py

import time
import pandas as pd
from model import load_model
from feature_extraction import extract_features
from data_collection import collect_sysmon_logs, preprocess_logs

def monitor_system_logs(log_path, model_path):
    """
    Continuously monitor the system logs and make real-time predictions.
    Args:
        log_path (str): Path to the log directory
        model_path (str): Path to the trained model
    """
    model = load_model(model_path)
    
    while True:
        logs = collect_sysmon_logs(log_path)
        logs = preprocess_logs(logs)
        features = extract_features(logs)
        
        # Predict activity
        predictions = model.predict(features)
        
        # Alert if malicious activity is detected
        for i, prediction in enumerate(predictions):
            if prediction == 'malicious':
                print(f"Alert: Malicious activity detected in log entry {i}")
        
        time.sleep(5)  # Check logs every 5 seconds

# Run the real-time monitoring (replace with your actual log path and model path)
monitor_system_logs('data/raw_logs', 'models/ransomware_detection_model.pkl')
