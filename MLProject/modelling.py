
import argparse
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


import mlflow

def main(data_path):
    print(f"📂 Memuat dataset dari: {data_path}")
    df = pd.read_csv(data_path)

    mlflow.set_experiment("EkaFanya_Workflow_CI")

    # ✅ Fix nested MLflow run (hindari error di GitHub Actions)
    if mlflow.active_run() is None:
        run_ctx = mlflow.start_run(run_name="RF_Workflow_CI")
    else:
        run_ctx = mlflow.active_run()

    with run_ctx:
        X_train, X_test, y_train, y_test = train_test_split(df["clean_text"], df["label"], test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_tfidf, y_train)

        preds = clf.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")

        print(f"🎯 Akurasi: {acc:.4f}")

    if mlflow.active_run():
        mlflow.end_run()
