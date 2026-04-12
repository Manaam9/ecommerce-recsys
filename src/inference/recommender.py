from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import polars as pl


@dataclass
class TwoStageRecommenderInference:
    als_model: object
    ranker_model: object
    user_item_matrix: object
    user2idx: dict
    idx2item: dict
    feature_cols: list[str]
    categorical_features: list[str]
    user_features_df: pl.DataFrame
    item_features_df: pl.DataFrame
    ui_features_df: pl.DataFrame | None = None
    user_col: str = "visitorid"
    item_col: str = "itemid"
    als_score_col: str = "als_score"

    def _model_features(self) -> list[str]:
        if hasattr(self.ranker_model, "booster_"):
            return list(self.ranker_model.booster_.feature_name())

        if hasattr(self.ranker_model, "feature_name"):
            return list(self.ranker_model.feature_name())

        return list(self.feature_cols)

    def _get_als_candidates(
        self,
        user_id,
        n_candidates: int = 100,
        filter_already_liked_items: bool = True,
    ) -> pd.DataFrame:
        if user_id not in self.user2idx:
            return pd.DataFrame(
                columns=[self.user_col, self.item_col, self.als_score_col]
            )

        user_idx = self.user2idx[user_id]
        user_items = self.user_item_matrix[user_idx]

        item_ids, scores = self.als_model.recommend(
            userid=user_idx,
            user_items=user_items,
            N=n_candidates,
            filter_already_liked_items=filter_already_liked_items,
        )

        rows = []
        for item_idx, score in zip(item_ids, scores):
            item_idx = int(item_idx)
            if item_idx in self.idx2item:
                rows.append(
                    {
                        self.user_col: user_id,
                        self.item_col: self.idx2item[item_idx],
                        self.als_score_col: float(score),
                    }
                )

        return pd.DataFrame(rows)

    def _build_context_features(
        self,
        candidates_pl: pl.DataFrame,
        ref_dt: datetime | None = None,
    ) -> pl.DataFrame:
        if ref_dt is None:
            ref_dt = datetime.now()

        ref_hour = ref_dt.hour
        ref_weekday = ref_dt.weekday()
        ref_is_weekend = int(ref_weekday >= 5)

        return candidates_pl.with_columns(
            pl.lit(ref_dt).alias("ref_timestamp_dt"),
            pl.lit(ref_hour).alias("ref_hour"),
            pl.lit(ref_weekday).alias("ref_weekday"),
            pl.lit(ref_is_weekend).alias("ref_is_weekend"),
        )

    def _add_missing_ui_defaults(self, df: pl.DataFrame) -> pl.DataFrame:
        model_features = set(self._model_features())

        zero_cols = [
            "ui_total_events",
            "ui_n_views",
            "ui_n_carts",
            "ui_n_transactions",
        ]

        for col in zero_cols:
            if col in model_features and col not in df.columns:
                df = df.with_columns(pl.lit(0).alias(col))

        if (
            "ui_recency_hours" in model_features
            and "ui_recency_hours" not in df.columns
        ):
            df = df.with_columns(pl.lit(9999.0).alias("ui_recency_hours"))

        return df

    def _join_features(
        self,
        candidates_df: pd.DataFrame,
        ref_dt: datetime | None = None,
    ) -> pl.DataFrame:
        if candidates_df.empty:
            return pl.DataFrame(
                schema={
                    self.user_col: pl.Int64,
                    self.item_col: pl.Int64,
                    self.als_score_col: pl.Float64,
                }
            )

        candidates_pl = pl.from_pandas(candidates_df)
        candidates_pl = self._build_context_features(candidates_pl, ref_dt=ref_dt)

        features_df = candidates_pl.join(
            self.user_features_df, on=self.user_col, how="left"
        ).join(self.item_features_df, on=self.item_col, how="left")

        if self.ui_features_df is not None:
            features_df = features_df.join(
                self.ui_features_df,
                on=[self.user_col, self.item_col],
                how="left",
            )

        features_df = self._add_missing_ui_defaults(features_df)

        fill_zero_cols = [
            "ui_total_events",
            "ui_n_views",
            "ui_n_carts",
            "ui_n_transactions",
            "user_total_events",
            "user_unique_items",
            "user_n_views",
            "user_n_carts",
            "user_n_transactions",
            "item_total_events",
            "item_unique_users",
            "item_n_views",
            "item_n_carts",
            "item_n_transactions",
            "item_prop_count",
            "item_unique_prop_count",
            "item_cart_rate",
            "item_buy_rate",
            "user_cart_rate",
            "user_buy_rate",
        ]

        existing_fill_zero_cols = [
            c for c in fill_zero_cols if c in features_df.columns
        ]
        if existing_fill_zero_cols:
            features_df = features_df.with_columns(
                [pl.col(c).fill_null(0) for c in existing_fill_zero_cols]
            )

        fill_big_cols = [
            "ui_recency_hours",
            "item_recency_hours",
            "user_recency_hours",
        ]
        existing_fill_big_cols = [c for c in fill_big_cols if c in features_df.columns]
        if existing_fill_big_cols:
            features_df = features_df.with_columns(
                [pl.col(c).fill_null(9999.0) for c in existing_fill_big_cols]
            )

        return features_df

    def _ensure_feature_columns(self, features_df: pl.DataFrame) -> pl.DataFrame:
        model_features = self._model_features()
        missing_cols = [c for c in model_features if c not in features_df.columns]

        if missing_cols:
            raise ValueError(f"Missing features: {missing_cols}")

        return features_df

    def _prepare_ranker_matrix(self, features_df: pl.DataFrame) -> pd.DataFrame:
        model_features = self._model_features()
        x = features_df.select(model_features).to_pandas()

        for col in self.categorical_features:
            if col in x.columns:
                x[col] = x[col].astype("category")

        return x

    def recommend(
        self,
        user_id,
        n_candidates: int = 100,
        top_k: int = 10,
        ref_dt: datetime | None = None,
        filter_already_liked_items: bool = True,
    ) -> pl.DataFrame:
        candidates_df = self._get_als_candidates(
            user_id=user_id,
            n_candidates=n_candidates,
            filter_already_liked_items=filter_already_liked_items,
        )

        if candidates_df.empty:
            return pl.DataFrame(
                schema={
                    self.user_col: pl.Int64,
                    self.item_col: pl.Int64,
                    self.als_score_col: pl.Float64,
                    "rerank_score": pl.Float64,
                }
            )

        features_df = self._join_features(candidates_df, ref_dt=ref_dt)
        features_df = self._ensure_feature_columns(features_df)

        x = self._prepare_ranker_matrix(features_df)
        rerank_scores = self.ranker_model.predict(x)

        result_df = features_df.with_columns(
            pl.Series("rerank_score", rerank_scores)
        ).sort("rerank_score", descending=True)

        return result_df.head(top_k)
