import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

EVENTS_PATH = os.path.join(RAW_DIR, "events.csv")
CATEGORY_TREE_PATH = os.path.join(RAW_DIR, "category_tree.csv")
ITEM_PROPERTIES_PATH_1 = os.path.join(RAW_DIR, "item_properties_part1.csv")
ITEM_PROPERTIES_PATH_2 = os.path.join(RAW_DIR, "item_properties_part2.csv")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# mlflow
MLFLOW_BASE_DIR = os.path.join(BASE_DIR, "mlflow")
MLFLOW_DIR = os.path.join(MLFLOW_BASE_DIR, "mlruns")
MLFLOW_DB_PATH = os.path.join(MLFLOW_BASE_DIR, "mlflow.db")

# airflow
AIRFLOW_DIR = os.path.join(BASE_DIR, "airflow")
AIRFLOW_DAGS_DIR = os.path.join(AIRFLOW_DIR, "dags")

TOP_K = 10
ALS_FACTORS = 64
ALS_ITERATIONS = 15
ALS_REGULARIZATION = 0.01
RANDOM_STATE = 42
