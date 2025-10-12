import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    # Setup MLflow tracking
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("ci_training")

    # Start MLflow run
    with mlflow.start_run():
        # Load dataset asli hasil preprocessing
        df = pd.read_csv("namadataset_preprocessing.csv")

        # Pisahkan fitur dan label
        X = df.drop("label", axis=1)
        y = df["label"]

        # Split data latih dan uji
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Buat dan latih model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Evaluasi
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc}")

        # Logging ke MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print("✅ Model training & logging completed successfully using real dataset!")
