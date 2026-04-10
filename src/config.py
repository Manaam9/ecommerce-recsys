import os

# Базовая директория проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------
# DATA PATHS
# ----------------------

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Основные файлы
EVENTS_PATH = os.path.join(RAW_DIR, "events.csv")
CATEGORY_TREE_PATH = os.path.join(RAW_DIR, "category_tree.csv")
ITEM_PROPERTIES_PATH = os.path.join(RAW_DIR, "item_properties.csv")

# ----------------------
# ARTIFACTS / MODELS
# ----------------------

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

ALS_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "als_model.pkl")
POPULAR_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "top_popular.pkl")

# ----------------------
# MLFLOW
# ----------------------

MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns")

# ----------------------
# PARAMETERS
# ----------------------

TOP_K = 10
ALS_FACTORS = 64
ALS_ITERATIONS = 15
ALS_REGULARIZATION = 0.01

# ----------------------
# RANDOM SEED
# ----------------------

RANDOM_STATE = 42
