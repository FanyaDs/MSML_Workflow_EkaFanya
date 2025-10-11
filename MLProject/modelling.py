import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def main(data_path):
    # Muat dataset
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['clean_text', 'sentiment'])
    X = df['clean_text'].astype(str)
    y = df['sentiment'].astype(str)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Konfigurasi MLflow
    mlflow.set_experiment("EkaFanya_Workflow_CI")
    mlflow.sklearn.autolog()

    # ⚙️ Tidak membuka start_run() manual — biarkan MLflow CLI yang kelola
    rf.fit(X_train_tfidf, y_train)
    preds = rf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)

    # Manual logging (tidak perlu start_run)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_features_tfidf", 5000)
    mlflow.log_metric("accuracy", acc)

    print(f"🎯 Accuracy: {acc:.4f}")
    print("📊 Classification Report:")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
