import os

# корень проекта: ecommerce-recsys/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# data
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# исходные файлы
EVENTS_PATH = os.path.join(RAW_DIR, "events.csv")
CATEGORY_TREE_PATH = os.path.join(RAW_DIR, "category_tree.csv")
ITEM_PROPERTIES_PATH = os.path.join(RAW_DIR, "item_properties.csv")

# артефакты
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")

ALS_MODEL_PATH = os.path.join(MODELS_DIR, "als_model.pkl")
POPULAR_MODEL_PATH = os.path.join(MODELS_DIR, "top_popular.pkl")

# mlflow
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns")

# параметры
TOP_K = 10
ALS_FACTORS = 64
ALS_ITERATIONS = 15
ALS_REGULARIZATION = 0.01

RANDOM_STATE = 42
