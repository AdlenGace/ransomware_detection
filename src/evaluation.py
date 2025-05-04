# evaluation.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test data features
        y_test (pd.Series): Test data labels
    Returns:
        None
    """
    y_pred = model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='binary'):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='binary'):.2f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='binary'):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
