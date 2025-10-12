# ==========================================================
# modelling_tuning.py — Advanced Level (DagsHub + Manual Logging + Artefak Tambahan)
# ==========================================================
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import mlflow
import mlflow.sklearn
import dagshub

# ==========================================================
# 1️⃣ Inisialisasi DagsHub
# ==========================================================
dagshub.init(
    repo_owner="FanyaDs",
    repo_name="MSML_EkaFanyaYohanaDs",
    mlflow=True
)

mlflow.set_tracking_uri("https://dagshub.com/FanyaDs/MSML_EkaFanyaYohanaDs.mlflow")
mlflow.set_experiment("EkaFanya_RF_Advanced")

# ==========================================================
# 2️⃣ Load Dataset
# ==========================================================
DATA_PATH = "/content/drive/MyDrive/SMSML_Eka-Fanya-Yohana-Dasilva/Membangun_model/namadataset_preprocessing/namadataset_preprocessing.csv"
df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

# Pastikan tidak ada missing value
df = df.dropna(subset=['clean_text', 'sentiment'])
X = df['clean_text'].astype(str)
y = df['sentiment'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================================
# 3️⃣ TF-IDF Vectorizer
# ==========================================================
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Simpan vocabulary TF-IDF
vocab_path = "tfidf_vocabulary.json"
with open(vocab_path, "w") as f:
    json.dump({k: int(v) for k, v in tfidf.vocabulary_.items()}, f)

# ==========================================================
# 4️⃣ Hyperparameter Tuning
# ==========================================================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 30, 50],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)
grid.fit(X_train_tfidf, y_train)
best_model = grid.best_estimator_

print("✅ Best Params:", grid.best_params_)

# ==========================================================
# 5️⃣ Evaluasi Model
# ==========================================================
y_pred = best_model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')

print(f"🎯 Akurasi: {acc:.4f}")
print(f"📊 F1 Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==========================================================
# 6️⃣ Visualisasi Confusion Matrix
# ==========================================================
cm = confusion_matrix(y_test, y_pred, labels=y.unique())
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=y.unique(), yticklabels=y.unique())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix — Random Forest Advanced")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ==========================================================
# 7️⃣ Logging ke MLflow (Manual)
# ==========================================================
with mlflow.start_run(run_name="RF_Advanced_ManualLog"):
    # Log parameter terbaik
    mlflow.log_params(grid.best_params_)

    # Log metrik performa
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("n_features_tfidf", len(tfidf.vocabulary_))

    # Log artefak tambahan
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(vocab_path)

    # Log model ke DagsHub (cloud)
    mlflow.sklearn.log_model(best_model, "model")

print("✅ Model, metrik, dan artefak berhasil di-log ke DagsHub MLflow Cloud!")
