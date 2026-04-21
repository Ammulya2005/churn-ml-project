import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import subprocess
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Fix MLflow path issue (Windows)
# ---------------------------
mlflow.set_tracking_uri("file:///C:/churn-ml-project/mlruns")
print("Tracking URI:", mlflow.get_tracking_uri())
os.makedirs("mlruns", exist_ok=True)

# ---------------------------
# Get dataset version (Git hash)
# ---------------------------
def get_dvc_version():
    try:
        version = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
        return version
    except:
        return "unknown"

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_csv("Telecom Customer Churn.csv")

# Cleaning
df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df = df.dropna()

# Encoding
df = pd.get_dummies(df, drop_first=True)

# Split
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# MLflow Experiment
# ---------------------------
mlflow.set_experiment("Churn_Model_Comparison")

with mlflow.start_run():

    # 🔗 Link dataset version to experiment
    dvc_version = get_dvc_version()
    mlflow.log_param("data_version", dvc_version)

    # Model
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Logging
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)

    # Save model for DVC
    joblib.dump(model, "model.pkl")

    # Log model in MLflow
    mlflow.sklearn.log_model(model, "model")

    # (Optional but good for marks)
    mlflow.log_artifact("Telecom Customer Churn.csv.dvc")

    print(f"Accuracy: {acc}")
