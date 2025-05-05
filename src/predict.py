import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Paths to model and scaler
model_path = '/home/adlen/Desktop/Projects/ransomware_detections_project/data/models/random_forest_model.pkl'
scaler_path = '/home/adlen/Desktop/Projects/ransomware_detections_project/data/models/scaler.pkl'

# Load the trained model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load new data to predict
new_data_path = '/home/adlen/Desktop/Projects/ransomware_detections_project/data/processed_data/new_input.csv'
new_data = pd.read_csv(new_data_path)

print(f"New data columns: {new_data.columns}")

# One-hot encode 'protocol' as done in training
new_data = pd.get_dummies(new_data, columns=['protocol'])

# Drop 'ip_address' if present
if 'ip_address' in new_data.columns:
    new_data = new_data.drop(columns=['ip_address'])

# Trained model expects these exact columns
trained_columns = ['packet_count', 'count', 'protocol_TCP', 'protocol_UDP']

# Add any missing columns with value 0
for col in trained_columns:
    if col not in new_data.columns:
        print(f"Adding missing column: {col}")
        new_data[col] = 0

# Ensure all extra columns are dropped
new_data = new_data[trained_columns]  # This both filters and reorders columns

# Check final columns before scaling
print(f"Final input columns to scaler/model: {new_data.columns.tolist()}")

# Scale features
X_new_scaled = scaler.transform(new_data)

# Predict
predictions = model.predict(X_new_scaled)

# Output results
for i, pred in enumerate(predictions):
    print(f"Sample {i+1} prediction: {'Ransomware' if pred == 1 else 'Benign'}")
