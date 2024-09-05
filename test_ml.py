import pytest
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
import pandas as pd
import numpy as np  # Make sure to import NumPy
from sklearn.linear_model import LogisticRegression

# Sample data for testing
data = pd.read_csv('./data/census.csv')

# Define categorical features outside of functions to avoid formatting issues
cat_features = [
    "workclass", "education", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country"
]

# Test 1: Check if the processed data returns the expected structure and type
def test_process_data():
    X_train, y_train, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    assert isinstance(X_train, np.ndarray), "X_train is not a NumPy array"
    assert isinstance(y_train, np.ndarray), "y_train is not a NumPy array"
    print("Processed X type:", type(X_train))  # Debugging step to check type of X
    print("Processed X shape:", X_train.shape)  # Debugging step to check shape of X
    print("--- End of Test 1 ---\n")

# Test 2: Verify that the trained model is of the correct type (LogisticRegression)
def test_train_model():
    X_train, y_train, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression), "The model is not an instance of LogisticRegression"
    print("Model type:", type(model))
    print("--- End of Test 2 ---\n")

# Test 3: Check if the compute_model_metrics function returns values within expected ranges
def test_compute_model_metrics():
    X_train, y_train, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    precision, recall, f1 = compute_model_metrics(y_train, preds)
    assert precision >= 0.0 and precision <= 1.0, "Precision out of expected range"
    assert recall >= 0.0 and recall <= 1.0, "Recall out of expected range"
    assert f1 >= 0.0 and f1 <= 1.0, "F1 score out of expected range"
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    print("--- End of Test 3 ---\n")

if __name__ == "__main__":
    test_process_data()
    test_train_model()
    test_compute_model_metrics()
