import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def load_data():
    print("Loading data...")
    # TODO: загрузка данных (S3 / локально)


def preprocess_data():
    print("Preprocessing data...")
    # TODO: очистка, фильтрация


def build_features():
    print("Building features...")
    # TODO: генерация фичей


def train_model():
    print("Training model...")
    subprocess.run(["python", "scripts/train_model.py"], check=True)


def save_model():
    print("Saving model...")
    # TODO: сохранение модели (MLflow / S3)


with DAG(
    dag_id="retrain_recsys_model",
    default_args=default_args,
    description="Retrain recommender system",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    t2 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    t3 = PythonOperator(
        task_id="build_features",
        python_callable=build_features,
    )

    t4 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    t5 = PythonOperator(
        task_id="save_model",
        python_callable=save_model,
    )

    t1 >> t2 >> t3 >> t4 >> t5
