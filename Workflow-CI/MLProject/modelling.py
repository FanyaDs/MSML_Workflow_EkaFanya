import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# ======================================
# 1️⃣ Setup MLflow Experiment
# ======================================
mlflow.set_experiment("MSML_Workflow_Fanya")

# ======================================
# 2️⃣ Load Dataset
# ======================================
base_path = os.path.dirname(__file__)
dataset_path = os.path.join(base_path, "namadataset_preprocessing", "namadataset_preprocessing.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset tidak ditemukan di {dataset_path}")

df = pd.read_csv(dataset_path)

if "label" not in df.columns:
    raise ValueError("❌ Kolom 'label' tidak ditemukan di dataset!")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================================
# 3️⃣ Training Model
# ======================================
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # ======================================
    # 4️⃣ Logging ke MLflow
    # ======================================
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"🎯 Akurasi model: {acc:.4f}")
    print("✅ Model berhasil dilatih dan disimpan di MLflow.")
