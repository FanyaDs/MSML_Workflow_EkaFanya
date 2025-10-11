
import argparse
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main(data_path):
    print(f"📂 Memuat dataset dari: {data_path}")
    df = pd.read_csv(data_path)
    X = df["clean_text"].astype(str)
    y = df["sentiment"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vec = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vec.fit_transform(X_train)
    X_test_tfidf = vec.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    mlflow.set_experiment("EkaFanya_Workflow_CI")

    with mlflow.start_run(run_name="RF_Workflow_CI"):
        clf.fit(X_train_tfidf, y_train)
        preds = clf.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        print(f"🎯 Akurasi: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/content/drive/MyDrive/SMSML_Eka-Fanya-Yohana-Dasilva/Workflow-CI/MLProject/namadataset_preprocessing/namadataset_preprocessing.csv")
    args = parser.parse_args()
    main(args.data_path)
