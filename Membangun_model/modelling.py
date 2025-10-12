# ==========================================================
# modelling.py — Kriteria 2 (Basic)
# Project: SMSML_Eka-Fanya-Yohana-Dasilva
# Deskripsi:
#   Melatih model Machine Learning (Random Forest) menggunakan MLflow autolog.
#   Hasil pelatihan disimpan di MLflow Tracking UI (localhost:5000)
# ==========================================================

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================================
# 1️⃣ Muat dataset hasil preprocessing
# Pastikan file CSV berasal dari tahapan Eksperimen/Preprocessing
# ==========================================================
DATA_PATH = "/content/drive/MyDrive/SMSML_Eka-Fanya-Yohana-Dasilva/Membangun_model/namadataset_preprocessing/namadataset_preprocessing.csv"
df = pd.read_csv(DATA_PATH)

# Bersihkan data kosong dan ubah tipe data menjadi string
df = df.dropna(subset=["clean_text", "sentiment"])
df["clean_text"] = df["clean_text"].astype(str)
df["sentiment"] = df["sentiment"].astype(str)

# Fitur dan label
X = df["clean_text"]
y = df["sentiment"]

# ==========================================================
# 2️⃣ Split data menjadi train dan test
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================================
# 3️⃣ TF-IDF Vectorizer + Random Forest Classifier
# ==========================================================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

# ==========================================================
# 4️⃣ MLflow autolog: otomatis menyimpan semua artefak
# ==========================================================
mlflow.set_experiment("EkaFanya_RF_Basic")  # nama eksperimen di MLflow
mlflow.sklearn.autolog()  # aktifkan autologging

with mlflow.start_run(run_name="RandomForest_Autolog"):
    rf.fit(X_train_tfidf, y_train)
    preds = rf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)

    print(f"🎯 Akurasi: {acc:.4f}")
    print("\n📊 Classification Report:\n", classification_report(y_test, preds))

    print("\n✅ Model dan artefak tersimpan di MLflow Tracking UI (localhost:5000)")
