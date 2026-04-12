from typing import List

from pydantic import BaseModel


class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10
    n_candidates: int = 100


class RecommendationItem(BaseModel):
    item_id: int
    score: float
    als_score: float | None = None
    categoryid: int | None = None


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]
