import os
import pandas as pd
import joblib
import pytest
from sklearn.metrics import accuracy_score

@pytest.mark.parametrize("path", ["data/data.csv"])
def test_data_integrity(path):
    assert os.path.exists(path), f"{path} missing"
    df = pd.read_csv(path)
    assert not df.empty, "Dataset is empty"
    assert df.isna().sum().sum() == 0, "Dataset contains missing values"
    print(f" Data integrity test passed for: {path}")

def test_feature_schema():
    df = pd.read_csv("data/data.csv")
    expected_cols = {"sepal_length", "petal_length", "sepal_width", "petal_width", "species"}
    assert set(df.columns) == expected_cols, f"Unexpected schema: {set(df.columns)}"
    print(" Feature schema test passed")

def test_model_performance():
    model_path = "artifacts/model.joblib"
    data_path = "data/data.csv"
    assert os.path.exists(model_path), f"{model_path} missing"
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X = df.drop(columns=["species"])
    y = df["species"]
    preds = model.predict(X)
    accuracy = accuracy_score(y, preds) * 100
    assert accuracy > 90.0, f"Model accuracy too low: {accuracy:.2f}%"
    print(f" Model performance test passed with accuracy: {accuracy:.2f}%")
