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
<<<<<<< HEAD
new_data = pd.get_dummies(new_data, columns=['protocol'])
=======
new_data = pd.get_dummies(new_data, columns=['protocol'], drop_first=True)
>>>>>>> 78923650777db9890c90ae3e5d1d1c5be31163dc

# Drop 'ip_address' if present
if 'ip_address' in new_data.columns:
    new_data = new_data.drop(columns=['ip_address'])

<<<<<<< HEAD
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
=======
# List of columns the model was trained on (Update this with the exact trained columns)
trained_columns = ['packet_count', 'count', 'protocol_UDP', 'protocol_TCP']  # Ensure these match training columns

# Print the trained columns and new data columns for comparison
print(f"Trained columns: {trained_columns}")
print(f"New data columns: {new_data.columns}")

# Check if there are missing columns
missing_columns = set(trained_columns) - set(new_data.columns)
if missing_columns:
    print(f"Missing columns in new data: {missing_columns}")

# Ensure the column order and add missing columns with default values (0)
for col in trained_columns:
    if col not in new_data.columns:
        new_data[col] = 0

# Reorder the columns to match the training order
new_data = new_data[trained_columns]

# Apply the same scaler used during training
X_new_scaled = scaler.transform(new_data)

# Make predictions
predictions = model.predict(X_new_scaled)

# Output predictions
>>>>>>> 78923650777db9890c90ae3e5d1d1c5be31163dc
for i, pred in enumerate(predictions):
    print(f"Sample {i+1} prediction: {'Ransomware' if pred == 1 else 'Benign'}")
