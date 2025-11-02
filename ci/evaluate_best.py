import os, json, pandas as pd, mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score

# -------- Config via env --------
MODEL_NAME = os.getenv("MODEL_NAME", "Iris-Classifier")
PREFERRED_STAGE = os.getenv("PREFERRED_STAGE", "Production")
BEST_METRIC = os.getenv("BEST_METRIC", "accuracy")
ART_SUBPATH = os.getenv("MODEL_ARTIFACT_SUBPATH", "iris_model")

# Option A (two files) - optional; used only if both exist
X_TEST_PATH = os.getenv("X_TEST_PATH", "")
Y_TEST_PATH = os.getenv("Y_TEST_PATH", "")

# Option B (single CSV) - default
DATA_CSV_PATH = os.getenv("DATA_CSV_PATH", "data/data.csv")
TARGET_COL = os.getenv("TARGET_COL", "species")  # change if your label column has a different name

def load_from_stage(name, stage):
    uri = f"models:/{name}/{stage}"
    model = mlflow.pyfunc.load_model(uri)
    return model, {"source": "stage", "uri": uri}

def load_best_by_metric(metric):
    client = MlflowClient()
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "")
    if exp_name:
        exp = client.get_experiment_by_name(exp_name)
        exp_ids = [exp.experiment_id] if exp else []
    else:
        exp_ids = [e.experiment_id for e in client.search_experiments()]

    best_run, best_val = None, float("-inf")
    for eid in exp_ids:
        runs = client.search_runs([eid], order_by=[f"metrics.{metric} DESC"], max_results=1)
        if runs:
            val = runs[0].data.metrics.get(metric, float("-inf"))
            if val > best_val:
                best_val, best_run = val, runs[0]
    if not best_run:
        raise RuntimeError("No runs found to select best-by-metric.")

    model_uri = f"runs:/{best_run.info.run_id}/{ART_SUBPATH}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model, {"source": "best-metric", "uri": model_uri, "run_id": best_run.info.run_id,
                   "metric": metric, "metric_value": best_val}

def load_eval_data():
    # If both X/Y files exist, use them
    if X_TEST_PATH and Y_TEST_PATH and os.path.exists(X_TEST_PATH) and os.path.exists(Y_TEST_PATH):
        X = pd.read_csv(X_TEST_PATH)
        y = pd.read_csv(Y_TEST_PATH).iloc[:, 0]
        return X, y, {"mode": "XY-files", "X": X_TEST_PATH, "y": Y_TEST_PATH}
    # Otherwise, use single CSV
    if not os.path.exists(DATA_CSV_PATH):
        raise FileNotFoundError(f"DATA_CSV_PATH not found: {DATA_CSV_PATH}")
    df = pd.read_csv(DATA_CSV_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL='{TARGET_COL}' not found in columns: {list(df.columns)}")
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y, {"mode": "single-csv", "data": DATA_CSV_PATH, "target_col": TARGET_COL, "n_rows": len(df)}

def main():
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # Load model
    try:
        model, meta = load_from_stage(MODEL_NAME, PREFERRED_STAGE) if PREFERRED_STAGE else load_best_by_metric(BEST_METRIC)
    except Exception:
        model, meta = load_best_by_metric(BEST_METRIC)

    # Load eval data
    X, y, data_meta = load_eval_data()

    # Predict
    y_pred = model.predict(X)

    # If pyfunc returns probabilities for classifiers, pick argmax
    try:
        import numpy as np
        if getattr(y_pred, "ndim", 1) == 2:
            y_pred = y_pred.argmax(axis=1)
    except Exception:
        pass

    # Compute accuracy (works with string labels too)
    acc = accuracy_score(y, y_pred)

    out = {
        "model_source": meta,
        "data_source": data_meta,
        "sanity_metrics": {"accuracy": acc}
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
