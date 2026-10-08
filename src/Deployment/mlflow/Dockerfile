# MLflow tracking server + model registry (Postgres backend store).
FROM python:3.11-slim
RUN pip install --no-cache-dir "mlflow>=2.16,<3" "sqlalchemy>=2.0,<2.1" psycopg2-binary
EXPOSE 5000
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" --artifacts-destination /mlartifacts --serve-artifacts"]
