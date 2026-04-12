from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "andrejmoldovan",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="retrain_ecommerce_recommender",
    default_args=default_args,
    description="Retraining pipeline for two-stage e-commerce recommender",
    schedule_interval="0 3 * * 1",
    start_date=datetime(2026, 4, 1),
    catchup=False,
    max_active_runs=1,
    tags=["recsys", "training", "ecommerce"],
) as dag:

    check_project_structure = BashOperator(
        task_id="check_project_structure",
        bash_command="""
        cd /opt/airflow/project && \
        test -f scripts/train_recommender_pipeline.py && \
        test -d models && \
        test -d data
        """,
    )

    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command="""
        cd /opt/airflow/project && \
        python scripts/train_recommender_pipeline.py --stage preprocess
        """,
    )

    train_als = BashOperator(
        task_id="train_als",
        bash_command="""
        cd /opt/airflow/project && \
        python scripts/train_recommender_pipeline.py --stage train_als
        """,
    )

    generate_candidates = BashOperator(
        task_id="generate_candidates",
        bash_command="""
        cd /opt/airflow/project && \
        python scripts/train_recommender_pipeline.py --stage generate_candidates
        """,
    )

    train_ranker = BashOperator(
        task_id="train_ranker",
        bash_command="""
        cd /opt/airflow/project && \
        python scripts/train_recommender_pipeline.py --stage train_ranker
        """,
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command="""
        cd /opt/airflow/project && \
        python scripts/train_recommender_pipeline.py --stage evaluate
        """,
    )

    save_artifacts = BashOperator(
        task_id="save_artifacts",
        bash_command="""
        cd /opt/airflow/project && \
        python scripts/train_recommender_pipeline.py --stage save_artifacts
        """,
    )

    (
        check_project_structure
        >> preprocess_data
        >> train_als
        >> generate_candidates
        >> train_ranker
        >> evaluate_model
        >> save_artifacts
    )
