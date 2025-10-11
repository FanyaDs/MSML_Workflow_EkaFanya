import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import os

def main(data_path):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['clean_text', 'sentiment'])
    X = df['clean_text'].astype(str)
    y = df['sentiment'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    mlflow.set_experiment("EkaFanya_Workflow_CI")
    mlflow.sklearn.autolog()

    rf.fit(X_train_tfidf, y_train)
    preds = rf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_features_tfidf", 5000)
    mlflow.log_metric("accuracy", acc)

    # Simpan model sebagai artefak (agar MLmodel tersedia untuk Docker)
    model_dir = "mlruns_model"
    os.makedirs(model_dir, exist_ok=True)
    mlflow.sklearn.log_model(rf, "model")
    mlflow.sklearn.save_model(rf, model_dir)

    print(f"🎯 Accuracy: {acc:.4f}")
    print("📊 Classification Report:")
    print(classification_report(y_test, preds))
    print(f"✅ Model berhasil disimpan ke {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
