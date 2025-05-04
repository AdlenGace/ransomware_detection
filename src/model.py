import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import accuracy_score, classification_report

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model.
    Args:
        X_train (pd.DataFrame): Training data features
        y_train (pd.Series): Training data labels
    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """
    Train a Support Vector Machine model.
    Args:
        X_train (pd.DataFrame): Training data features
        y_train (pd.Series): Training data labels
    Returns:
        SVC: Trained model
    """
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_mlp(X_train, y_train):
    """
    Train a Multilayer Perceptron model.
    Args:
        X_train (pd.DataFrame): Training data features
        y_train (pd.Series): Training data labels
    Returns:
        MLPClassifier: Trained model
    """
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    """
    Save the trained model to a file.
    Args:
        model: Trained model
        model_path (str): Path where the model should be saved
    """
    joblib.dump(model, model_path)

def load_model(model_path):
    """
    Load a trained model from a file.
    Args:
        model_path (str): Path to the saved model
    Returns:
        Model: Loaded model
    """
    return joblib.load(model_path)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test data features
        y_test (pd.Series): Test data labels
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def main():
    """
    Main function to train models, save them, and evaluate.
    """
    # Load data
    data_path = '/home/adlen/Desktop/Projects/ransomware_detections_project/data/processed_data/extracted_features.csv'
    data = pd.read_csv(data_path)
    
    # Handle missing data if needed
    data.fillna(0, inplace=True)  # Filling NaN values with 0 for simplicity

    # Check if the 'label' column exists
    if 'label' not in data.columns:
        print("Warning: 'label' column not found in the data. Assigning default labels.")
        data['label'] = 0  # Assign all 0 or customize this logic as needed

    # One-Hot Encode the 'protocol' column
    data = pd.get_dummies(data, columns=['protocol'], drop_first=True)

    # Split features and labels
    X = data.drop(columns=['ip_address', 'label'])
    y = data['label']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the fitted scaler
    scaler_path = '/home/adlen/Desktop/Projects/ransomware_detections_project/data/models/scaler.pkl'
    joblib.dump(scaler, scaler_path)

    # Train the model
    model_path = '/home/adlen/Desktop/Projects/ransomware_detections_project/data/models/random_forest_model.pkl'
    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, model_path)

    # Evaluate the model
    evaluate_model(rf_model, X_test, y_test)

if __name__ == "__main__":
    main()
