from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"
USER_COL = "visitorid"
ITEM_COL = "itemid"
TIME_COL = "ref_timestamp_dt"

TOP_K = 10
ALS_TOP_N = 100
RANDOM_STATE = 42

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


def load_events() -> pl.DataFrame:
    path = PROCESSED_DIR / "events.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pl.read_parquet(path)

    if TIME_COL in df.columns and df.schema[TIME_COL] != pl.Datetime:
        if df.schema[TIME_COL] == pl.Utf8:
            df = df.with_columns(
                pl.col(TIME_COL).str.strptime(pl.Datetime, strict=False).alias(TIME_COL)
            )
        else:
            df = df.with_columns(pl.col(TIME_COL).cast(pl.Datetime).alias(TIME_COL))

    return df


def temporal_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, object]:
    ts_sorted = df.select(TIME_COL).drop_nulls().sort(TIME_COL).to_series()
    if len(ts_sorted) == 0:
        raise ValueError(f"Column {TIME_COL} is empty")

    cutoff_idx = int(len(ts_sorted) * 0.8)
    cutoff_ts = ts_sorted[cutoff_idx]

    train_df = df.filter(pl.col(TIME_COL) < cutoff_ts)
    valid_df = df.filter(pl.col(TIME_COL) >= cutoff_ts)

    if train_df.height == 0 or valid_df.height == 0:
        raise ValueError("Temporal split produced empty train or valid")

    return train_df, valid_df, cutoff_ts


def preprocess() -> None:
    events = load_events()
    output_path = PROCESSED_DIR / "events_prepared.parquet"
    events.write_parquet(output_path)
    print(f"Saved preprocessed events -> {output_path}")


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
    item2idx = {item: idx for idx, item in idx2item.items()}

    return matrix, user2idx, item2idx, idx2item


def train_als_stage() -> None:
    events_path = PROCESSED_DIR / "events_prepared.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing preprocessed file: {events_path}")

    events = pl.read_parquet(events_path)
    train_events, _, _ = temporal_split(events)

    matrix, user2idx, item2idx, idx2item = build_interaction_matrix(train_events)

    als_model = AlternatingLeastSquares(
        factors=64,
        regularization=0.01,
        iterations=20,
        random_state=RANDOM_STATE,
    )
    als_model.fit(matrix)

    joblib.dump(als_model, MODELS_DIR / "als_model.bin")
    joblib.dump(matrix, MODELS_DIR / "user_item_matrix.pkl")
    joblib.dump(user2idx, MODELS_DIR / "user2idx.pkl")
    joblib.dump(item2idx, MODELS_DIR / "item2idx.pkl")
    joblib.dump(idx2item, MODELS_DIR / "idx2item.pkl")

    print("Saved ALS model and mappings")


def build_user_features(events: pl.DataFrame, cutoff_ts) -> pl.DataFrame:
    return (
        events.group_by(USER_COL)
        .agg(
            pl.len().alias("user_total_events"),
            pl.col(ITEM_COL).n_unique().alias("user_unique_items"),
            (
                pl.col("is_view").sum().alias("user_n_views")
                if "is_view" in events.columns
                else pl.lit(0).alias("user_n_views")
            ),
            (
                pl.col("is_cart").sum().alias("user_n_carts")
                if "is_cart" in events.columns
                else pl.lit(0).alias("user_n_carts")
            ),
            (
                pl.col("is_transaction").sum().alias("user_n_transactions")
                if "is_transaction" in events.columns
                else pl.lit(0).alias("user_n_transactions")
            ),
            pl.max(TIME_COL).alias("user_last_event_dt"),
        )
        .with_columns(
            (
                pl.col("user_n_carts") / pl.col("user_total_events").clip(lower_bound=1)
            ).alias("user_cart_rate"),
            (
                pl.col("user_n_transactions")
                / pl.col("user_total_events").clip(lower_bound=1)
            ).alias("user_buy_rate"),
            ((pl.lit(cutoff_ts) - pl.col("user_last_event_dt")).dt.total_hours()).alias(
                "user_recency_hours"
            ),
        )
    )


def build_item_features(events: pl.DataFrame, cutoff_ts) -> pl.DataFrame:
    cols = [ITEM_COL]
    if "available" in events.columns:
        cols.append("available")
    if "categoryid" in events.columns:
        cols.append("categoryid")
    if "parent_categoryid" in events.columns:
        cols.append("parent_categoryid")
    if "item_prop_count" in events.columns:
        cols.append("item_prop_count")
    if "item_unique_prop_count" in events.columns:
        cols.append("item_unique_prop_count")

    base = events.select(cols).unique(subset=[ITEM_COL])

    aggs = (
        events.group_by(ITEM_COL)
        .agg(
            pl.len().alias("item_total_events"),
            pl.col(USER_COL).n_unique().alias("item_unique_users"),
            (
                pl.col("is_view").sum().alias("item_n_views")
                if "is_view" in events.columns
                else pl.lit(0).alias("item_n_views")
            ),
            (
                pl.col("is_cart").sum().alias("item_n_carts")
                if "is_cart" in events.columns
                else pl.lit(0).alias("item_n_carts")
            ),
            (
                pl.col("is_transaction").sum().alias("item_n_transactions")
                if "is_transaction" in events.columns
                else pl.lit(0).alias("item_n_transactions")
            ),
            pl.max(TIME_COL).alias("item_last_event_dt"),
        )
        .with_columns(
            (
                pl.col("item_n_carts") / pl.col("item_total_events").clip(lower_bound=1)
            ).alias("item_cart_rate"),
            (
                pl.col("item_n_transactions")
                / pl.col("item_total_events").clip(lower_bound=1)
            ).alias("item_buy_rate"),
            ((pl.lit(cutoff_ts) - pl.col("item_last_event_dt")).dt.total_hours()).alias(
                "item_recency_hours"
            ),
        )
    )

    return base.join(aggs, on=ITEM_COL, how="left")


def build_ui_features(events: pl.DataFrame, cutoff_ts) -> pl.DataFrame:
    return (
        events.group_by([USER_COL, ITEM_COL])
        .agg(
            pl.len().alias("ui_total_events"),
            (
                pl.col("is_view").sum().alias("ui_n_views")
                if "is_view" in events.columns
                else pl.lit(0).alias("ui_n_views")
            ),
            (
                pl.col("is_cart").sum().alias("ui_n_carts")
                if "is_cart" in events.columns
                else pl.lit(0).alias("ui_n_carts")
            ),
            (
                pl.col("is_transaction").sum().alias("ui_n_transactions")
                if "is_transaction" in events.columns
                else pl.lit(0).alias("ui_n_transactions")
            ),
            pl.max(TIME_COL).alias("ui_last_event_dt"),
        )
        .with_columns(
            ((pl.lit(cutoff_ts) - pl.col("ui_last_event_dt")).dt.total_hours()).alias(
                "ui_recency_hours"
            ),
        )
    )


def generate_als_candidates_for_users(
    user_ids: list,
    als_model,
    user2idx: dict,
    user_item_matrix,
    idx2item: dict,
    n_candidates: int = ALS_TOP_N,
) -> pl.DataFrame:
    rows = []

    for user_id in user_ids:
        if user_id not in user2idx:
            continue

        user_idx = user2idx[user_id]
        user_items = user_item_matrix[user_idx]

        item_ids, scores = als_model.recommend(
            userid=user_idx,
            user_items=user_items,
            N=n_candidates,
            filter_already_liked_items=False,
        )

        for item_idx, score in zip(item_ids, scores):
            item_idx = int(item_idx)
            if item_idx in idx2item:
                rows.append(
                    {
                        USER_COL: user_id,
                        ITEM_COL: idx2item[item_idx],
                        "als_score": float(score),
                    }
                )

    if not rows:
        return pl.DataFrame(
            schema={USER_COL: pl.Int64, ITEM_COL: pl.Int64, "als_score": pl.Float64}
        )

    return pl.from_dicts(rows)


def generate_valid_targets(valid_events: pl.DataFrame) -> pl.DataFrame:
    positive = valid_events.filter(pl.col(TARGET_COL) > 0)
    return (
        positive.select([USER_COL, ITEM_COL])
        .unique()
        .with_columns(pl.lit(1).alias(TARGET_COL))
    )


def generate_candidates_stage() -> None:
    events_path = PROCESSED_DIR / "events_prepared.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing preprocessed file: {events_path}")

    events = pl.read_parquet(events_path)
    train_events, valid_events, cutoff_ts = temporal_split(events)

    als_model = joblib.load(MODELS_DIR / "als_model.bin")
    user_item_matrix = joblib.load(MODELS_DIR / "user_item_matrix.pkl")
    user2idx = joblib.load(MODELS_DIR / "user2idx.pkl")
    idx2item = joblib.load(MODELS_DIR / "idx2item.pkl")

    valid_users = valid_events.select(USER_COL).unique().to_series().to_list()

    candidates_df = generate_als_candidates_for_users(
        user_ids=valid_users,
        als_model=als_model,
        user2idx=user2idx,
        user_item_matrix=user_item_matrix,
        idx2item=idx2item,
        n_candidates=ALS_TOP_N,
    )

    valid_target = generate_valid_targets(valid_events)

    ranker_df = candidates_df.join(
        valid_target,
        on=[USER_COL, ITEM_COL],
        how="left",
    ).with_columns(
        pl.col(TARGET_COL).fill_null(0).cast(pl.Int8),
        pl.lit(cutoff_ts).alias(TIME_COL),
        pl.lit(cutoff_ts.hour).alias("ref_hour"),
        pl.lit(cutoff_ts.weekday()).alias("ref_weekday"),
        pl.lit(int(cutoff_ts.weekday() >= 5)).alias("ref_is_weekend"),
    )

    user_features_df = build_user_features(train_events, cutoff_ts)
    item_features_df = build_item_features(train_events, cutoff_ts)
    ui_features_df = build_ui_features(train_events, cutoff_ts)

    ranker_df = (
        ranker_df.join(user_features_df, on=USER_COL, how="left")
        .join(item_features_df, on=ITEM_COL, how="left")
        .join(ui_features_df, on=[USER_COL, ITEM_COL], how="left")
    )

    fill_zero_cols = [
        "ui_total_events",
        "ui_n_views",
        "ui_n_carts",
        "ui_n_transactions",
    ]
    existing = [c for c in fill_zero_cols if c in ranker_df.columns]
    if existing:
        ranker_df = ranker_df.with_columns([pl.col(c).fill_null(0) for c in existing])

    if "ui_recency_hours" in ranker_df.columns:
        ranker_df = ranker_df.with_columns(pl.col("ui_recency_hours").fill_null(9999.0))

    ranker_df.write_parquet(PROCESSED_DIR / "ranker_dataset.parquet")
    user_features_df.write_parquet(PROCESSED_DIR / "user_features.parquet")
    item_features_df.write_parquet(PROCESSED_DIR / "item_features.parquet")
    ui_features_df.write_parquet(PROCESSED_DIR / "ui_features.parquet")
    valid_target.write_parquet(PROCESSED_DIR / "valid_target.parquet")

    print("Saved ranker dataset and feature tables")


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

    train_pd = df.select(feature_cols + [TARGET_COL, USER_COL, ITEM_COL]).to_pandas()

    x_train = train_pd[feature_cols].copy()
    y_train = train_pd[TARGET_COL].astype(int).copy()

    for col in CATEGORICAL_FEATURES:
        if col in x_train.columns:
            x_train[col] = x_train[col].astype("category")

    group_train = train_pd.groupby(USER_COL).size().tolist()

    return train_pd, x_train, y_train, group_train, feature_cols


def train_ranker_stage() -> None:
    ranker_path = PROCESSED_DIR / "ranker_dataset.parquet"
    if not ranker_path.exists():
        raise FileNotFoundError(f"Missing ranker dataset: {ranker_path}")

    ranker_dataset = pl.read_parquet(ranker_path)
    _, x_train, y_train, group_train, feature_cols = prepare_ranker_data(ranker_dataset)

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[TOP_K],
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
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        x_train,
        y_train,
        group=group_train,
        eval_at=[TOP_K],
        categorical_feature=[c for c in CATEGORICAL_FEATURES if c in x_train.columns],
    )

    joblib.dump(model, MODELS_DIR / "lgbm_ranker.bin")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")

    print("Saved ranker model")


def recall_at_k(actual: list, predicted: list, k: int = 10) -> float:
    predicted_k = predicted[:k]
    actual_set = set(actual)
    if len(actual_set) == 0:
        return 0.0
    return len(actual_set.intersection(predicted_k)) / len(actual_set)


def ndcg_at_k(actual: list, predicted: list, k: int = 10) -> float:
    predicted = predicted[:k]
    actual_set = set(actual)

    if len(actual_set) == 0:
        return 0.0

    dcg = 0.0
    for i, p in enumerate(predicted, start=1):
        if p in actual_set:
            dcg += 1.0 / np.log2(i + 1)

    ideal_hits = min(len(actual_set), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_stage() -> None:
    ranker_dataset = pl.read_parquet(PROCESSED_DIR / "ranker_dataset.parquet")
    valid_target = pl.read_parquet(PROCESSED_DIR / "valid_target.parquet")

    model = joblib.load(MODELS_DIR / "lgbm_ranker.bin")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

    df = ranker_dataset.to_pandas()
    x = df[feature_cols].copy()

    for col in CATEGORICAL_FEATURES:
        if col in x.columns:
            x[col] = x[col].astype("category")

    preds = model.predict(x)
    df["score"] = preds

    actual_map = (
        valid_target.to_pandas().groupby(USER_COL)[ITEM_COL].apply(list).to_dict()
    )

    recalls = []
    ndcgs = []

    for user_id, group in df.groupby(USER_COL):
        if user_id not in actual_map:
            continue

        predicted_items = group.sort_values("score", ascending=False)[ITEM_COL].tolist()
        actual_items = actual_map[user_id]

        recalls.append(recall_at_k(actual_items, predicted_items, k=TOP_K))
        ndcgs.append(ndcg_at_k(actual_items, predicted_items, k=TOP_K))

    metrics = {
        f"Recall@{TOP_K}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{TOP_K}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_users_evaluated": int(len(recalls)),
    }

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        import json

        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"Saved metrics -> {metrics_path}")
    print(metrics)


def save_artifacts_stage() -> None:
    user_item_matrix = joblib.load(MODELS_DIR / "user_item_matrix.pkl")
    user2idx = joblib.load(MODELS_DIR / "user2idx.pkl")
    idx2item = joblib.load(MODELS_DIR / "idx2item.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

    user_features_df = pl.read_parquet(PROCESSED_DIR / "user_features.parquet")
    item_features_df = pl.read_parquet(PROCESSED_DIR / "item_features.parquet")
    ui_features_df = pl.read_parquet(PROCESSED_DIR / "ui_features.parquet")

    inference_assets = {
        "user_item_matrix": user_item_matrix,
        "user2idx": user2idx,
        "idx2item": idx2item,
        "feature_cols": feature_cols,
        "categorical_features": CATEGORICAL_FEATURES,
        "user_features_df": user_features_df,
        "item_features_df": item_features_df,
        "ui_features_df": ui_features_df,
    }

    with open(MODELS_DIR / "inference_assets.pkl", "wb") as f:
        pickle.dump(inference_assets, f)

    print("Saved inference assets")


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
