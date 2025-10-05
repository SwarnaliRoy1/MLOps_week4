"""
train.py — Iris Classifier Training Script
------------------------------------------
This script trains a simple Decision Tree classifier on the Iris dataset,
saves the trained model to 'artifacts/model.joblib', and logs accuracy.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib, os, datetime

# Load dataset
DATA_PATH = "data/data.csv"
df = pd.read_csv(DATA_PATH)

#  Split data
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#  Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#  Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

#  Save model artifact with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
artifact_dir = f"artifacts/{timestamp}"
os.makedirs(artifact_dir, exist_ok=True)

model_path = f"{artifact_dir}/model.joblib"
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

#  Save metrics
metrics_path = f"{artifact_dir}/metrics.txt"
with open(metrics_path, "w") as f:
    f.write(f"accuracy: {accuracy:.4f}\n")

print(f"Metrics saved to: {metrics_path}")
