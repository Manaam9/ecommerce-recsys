#!/usr/bin/env bash

set -e

echo "Starting MLflow..."

# переходим в папку mlflow
mkdir -p mlflow
cd mlflow

# создаём папку для артефактов
mkdir -p mlruns

# запуск через python (чтобы не было проблем с PATH)
python -m mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
  