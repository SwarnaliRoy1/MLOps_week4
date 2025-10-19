import os
import pandas as pd

def test_data_integrity():
    """Check if main branch data is present"""
    assert os.path.exists("data/data.csv"), " data/data.csv missing in main branch"

def test_model_artifact():
    """Ensure model artifact was created after training"""
    assert os.path.exists("artifacts/model.joblib"), " Model artifact missing"

def test_metrics_csv():
    """Ensure metrics file exists and has correct format"""
    assert os.path.exists("artifacts/metrics.csv"), " Metrics CSV not found"
    df = pd.read_csv("artifacts/metrics.csv")
    assert "accuracy" in df.columns, " Accuracy column missing in metrics.csv"
    assert df["accuracy"].iloc[0] > 0.9, " Model accuracy seems too low"
