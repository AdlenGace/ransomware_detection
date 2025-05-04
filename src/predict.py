# predict.py

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model_path = '/home/adlen/Desktop/Projects/ransomware_detections_project/data/models/random_forest_model.pkl'
model = joblib.load(model_path)

# Load new data (the same format as your training data)
new_data_path = '/home/adlen/Desktop/Projects/ransomware_detections_project/data/new_data.csv'  # Adjust path as needed
new_data = pd.read_csv(new_data_path)

# Preprocess new data
# Assuming 'ip_address' and 'protocol' are not features and should be dropped
X_new = new_data.drop(columns=['ip_address', 'protocol'])

# Ensure that missing or non-numeric values are handled (convert 'protocol' to numeric if needed)
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Predict with the model
predictions = model.predict(X_new_scaled)

# Add predictions to the dataframe
new_data['prediction'] = predictions

# Save results to a new CSV
new_data.to_csv('/home/adlen/Desktop/Projects/ransomware_detections_project/data/predictions.csv', index=False)
print("Predictions saved to /home/adlen/Desktop/Projects/ransomware_detections_project/data/predictions.csv")
