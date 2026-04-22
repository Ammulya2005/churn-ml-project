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
mlflow.set_tracking_uri("sqlite:///mlflow.db")
print("Tracking URI:", mlflow.get_tracking_uri())
os.makedirs("mlruns", exist_ok=True)
def get_dvc_version():
    try:
        version = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
        return version
    except:
        return "unknown"
df = pd.read_csv("Telecom Customer Churn.csv")
df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
mlflow.set_experiment("Churn_Model_Comparison")

with mlflow.start_run():
    dvc_version = get_dvc_version()
    mlflow.log_param("data_version", dvc_version)
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    joblib.dump(model, "model.pkl")
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("Telecom Customer Churn.csv.dvc")

    print(f"Accuracy: {acc}")
