import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"
USER_COL = "visitorid"
ITEM_COL = "itemid"
TIME_COL = "ref_timestamp_dt"

CATEGORICAL_FEATURES = [
    "categoryid",
    "parent_categoryid",
    "available",
    "ref_hour",
    "ref_weekday",
    "ref_is_weekend",
]

DROP_COLS = [
    TARGET_COL,
    USER_COL,
    ITEM_COL,
    TIME_COL,
    "user_last_event_dt",
    "item_last_event_dt",
    "ui_last_event_dt",
    "itemid_right",
]


def load_raw_events() -> pl.DataFrame:
    path = DATA_DIR / "processed" / "events.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pl.read_parquet(path)


def preprocess() -> None:
    events = load_raw_events()

    if TIME_COL in events.columns and events.schema[TIME_COL] != pl.Datetime:
        if events.schema[TIME_COL] == pl.Utf8:
            events = events.with_columns(
                pl.col(TIME_COL).str.strptime(pl.Datetime, strict=False).alias(TIME_COL)
            )

    output_path = DATA_DIR / "processed" / "events_prepared.parquet"
    events.write_parquet(output_path)
    print(f"Saved preprocessed events to {output_path}")


def build_interaction_matrix(events: pl.DataFrame):
    interactions = events.select([USER_COL, ITEM_COL, TARGET_COL]).to_pandas()

    user_codes, unique_users = pd.factorize(interactions[USER_COL])
    item_codes, unique_items = pd.factorize(interactions[ITEM_COL])

    values = interactions[TARGET_COL].astype(float).values
    matrix = csr_matrix(
        (values, (user_codes, item_codes)),
        shape=(len(unique_users), len(unique_items)),
    )

    user2idx = {user: idx for idx, user in enumerate(unique_users)}
    idx2item = {idx: item for idx, item in enumerate(unique_items)}

    return matrix, user2idx, idx2item


def train_als_stage() -> None:
    events = pl.read_parquet(DATA_DIR / "processed" / "events_prepared.parquet")

    matrix, user2idx, idx2item = build_interaction_matrix(events)

    als_model = AlternatingLeastSquares(
        factors=64,
        regularization=0.01,
        iterations=20,
        random_state=42,
    )
    als_model.fit(matrix)

    joblib.dump(als_model, MODELS_DIR / "als_model.bin")
    joblib.dump(matrix, MODELS_DIR / "user_item_matrix.pkl")
    joblib.dump(user2idx, MODELS_DIR / "user2idx.pkl")
    joblib.dump(idx2item, MODELS_DIR / "idx2item.pkl")

    print("ALS model and mappings saved")


def generate_candidates_stage() -> None:
    print("Candidate generation stage started")
    print("Here you should generate ALS candidates and save ranker dataset")
    print("Example output: data/processed/ranker_dataset.parquet")


def prepare_ranker_data(ranker_dataset: pl.DataFrame):
    df = ranker_dataset.clone()

    if df.schema[TIME_COL] != pl.Datetime:
        if df.schema[TIME_COL] == pl.Utf8:
            df = df.with_columns(
                pl.col(TIME_COL).str.strptime(pl.Datetime, strict=False).alias(TIME_COL)
            )
        else:
            df = df.with_columns(pl.col(TIME_COL).cast(pl.Datetime).alias(TIME_COL))

    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    ts_sorted = df.select(TIME_COL).drop_nulls().sort(TIME_COL).to_series()
    cutoff_idx = int(len(ts_sorted) * 0.8)
    cutoff_ts = ts_sorted[cutoff_idx]

    train_df = df.filter(pl.col(TIME_COL) < cutoff_ts)
    valid_df = df.filter(pl.col(TIME_COL) >= cutoff_ts)

    train_pd = train_df.select(feature_cols + [TARGET_COL]).to_pandas()
    valid_pd = valid_df.select(feature_cols + [TARGET_COL]).to_pandas()

    x_train = train_pd[feature_cols].copy()
    y_train = train_pd[TARGET_COL].astype(int).copy()

    x_valid = valid_pd[feature_cols].copy()
    y_valid = valid_pd[TARGET_COL].astype(int).copy()

    for col in CATEGORICAL_FEATURES:
        if col in x_train.columns:
            x_train[col] = x_train[col].astype("category")
            x_valid[col] = x_valid[col].astype("category")

    return x_train, y_train, x_valid, y_valid, feature_cols


def train_ranker_stage() -> None:
    ranker_path = DATA_DIR / "processed" / "ranker_dataset.parquet"
    if not ranker_path.exists():
        raise FileNotFoundError(f"Missing ranker dataset: {ranker_path}")

    ranker_dataset = pl.read_parquet(ranker_path)
    x_train, y_train, x_valid, y_valid, feature_cols = prepare_ranker_data(
        ranker_dataset
    )

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10],
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1e-3,
        reg_lambda=1e-3,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    group_train = (
        ranker_dataset.filter(
            pl.col(TIME_COL)
            < ranker_dataset.select(TIME_COL)
            .drop_nulls()
            .sort(TIME_COL)
            .to_series()[int(ranker_dataset.height * 0.8)]
        )
        .select(USER_COL)
        .to_pandas()
        .groupby(USER_COL)
        .size()
        .tolist()
    )

    group_valid = (
        ranker_dataset.filter(
            pl.col(TIME_COL)
            >= ranker_dataset.select(TIME_COL)
            .drop_nulls()
            .sort(TIME_COL)
            .to_series()[int(ranker_dataset.height * 0.8)]
        )
        .select(USER_COL)
        .to_pandas()
        .groupby(USER_COL)
        .size()
        .tolist()
    )

    model.fit(
        x_train,
        y_train,
        group=group_train,
        eval_set=[(x_valid, y_valid)],
        eval_group=[group_valid],
        eval_at=[10],
        categorical_feature=[c for c in CATEGORICAL_FEATURES if c in x_train.columns],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    joblib.dump(model, MODELS_DIR / "lgbm_ranker.bin")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")
    print("Ranker saved")


def evaluate_stage() -> None:
    ranker_path = DATA_DIR / "processed" / "ranker_dataset.parquet"
    ranker_dataset = pl.read_parquet(ranker_path)

    x_train, y_train, x_valid, y_valid, feature_cols = prepare_ranker_data(
        ranker_dataset
    )
    model = joblib.load(MODELS_DIR / "lgbm_ranker.bin")

    preds = model.predict(x_valid)
    metric = float(np.mean(preds))

    metrics = {
        "mean_prediction": metric,
        "n_valid_rows": int(len(x_valid)),
    }

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print("Evaluation metrics saved")


def save_artifacts_stage() -> None:
    print("Artifacts are already saved into models/ and artifacts/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()

    if args.stage == "preprocess":
        preprocess()
    elif args.stage == "train_als":
        train_als_stage()
    elif args.stage == "generate_candidates":
        generate_candidates_stage()
    elif args.stage == "train_ranker":
        train_ranker_stage()
    elif args.stage == "evaluate":
        evaluate_stage()
    elif args.stage == "save_artifacts":
        save_artifacts_stage()
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
