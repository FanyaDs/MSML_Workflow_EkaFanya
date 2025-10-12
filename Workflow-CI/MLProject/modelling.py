import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    # Setup MLflow tracking
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("ci_training")

    # Start run explicitly to avoid 'Run not found' error
    with mlflow.start_run():
        # Load dataset
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc}")

        # Log metrics and model
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print("✅ Model training & logging completed successfully!")
