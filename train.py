"""
train.py — Iris Classifier Training Script
------------------------------------------
This script trains a simple Decision Tree classifier on the Iris dataset,
saves the trained model to 'artifacts/model.joblib', and logs accuracy.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib, os, datetime

# Load dataset

# import yaml
# params = yaml.safe_load(open("params.yaml"))
# df = pd.read_csv(params["data_path"])

df = pd.read_csv("data/data.csv")

#  Split data
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#  Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#  Evaluate model
y_pred = model.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

#  Save model artifact with timestamp
# timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# artifact_dir = f"artifacts/{timestamp}"
# os.makedirs(artifact_dir, exist_ok=True)

# model_path = f"{artifact_dir}/model.joblib"
model_path = f"artifacts/model.joblib"
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

#  Save metrics
# metrics_path = f"{artifact_dir}/metrics.txt"
# with open(metrics_path, "w") as f:
#    f.write(f"accuracy: {accuracy:.4f}\n")

# print(f"Metrics saved to: {metrics_path}")

# Save metrics as CSV
metrics_path = "artifacts/metrics.csv"
metrics_df = pd.DataFrame({
    "metric": ["accuracy", "precision", "recall", "f1_score"],
    "value": [accuracy, precision, recall, f1]
})

metrics_df.to_csv(metrics_path, index=False)
print(f"Metrics saved to: {metrics_path}")
