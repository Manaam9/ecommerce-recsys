import pickle
from pathlib import Path

import joblib

from src.inference.recommender import TwoStageRecommenderInference

MODELS_DIR = Path("models")
ALS_PATH = MODELS_DIR / "als_model.bin"
RANKER_PATH = MODELS_DIR / "lgbm_ranker.bin"
ASSETS_PATH = MODELS_DIR / "inference_assets.pkl"


def load_als_model():
    return joblib.load(ALS_PATH)


def load_ranker_model():
    return joblib.load(RANKER_PATH)


def load_assets():
    with open(ASSETS_PATH, "rb") as f:
        return pickle.load(f)


def load_inference_model() -> TwoStageRecommenderInference:
    als_model = load_als_model()
    ranker_model = load_ranker_model()
    assets = load_assets()

    return TwoStageRecommenderInference(
        als_model=als_model,
        ranker_model=ranker_model,
        user_item_matrix=assets["user_item_matrix"],
        user2idx=assets["user2idx"],
        idx2item=assets["idx2item"],
        feature_cols=assets["feature_cols"],
        categorical_features=assets["categorical_features"],
        user_features_df=assets["user_features_df"],
        item_features_df=assets["item_features_df"],
        ui_features_df=assets.get("ui_features_df"),
    )
