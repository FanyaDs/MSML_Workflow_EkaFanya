# ===== Custom Dockerfile for MLflow Model =====
FROM python:3.12-slim

WORKDIR /app

COPY MLProject/mlruns_model /app/model

RUN pip install --upgrade pip     && pip install mlflow scikit-learn pandas

EXPOSE 5000

CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "5000"]
