from fastapi import FastAPI, HTTPException

from src.api.dependencies import load_inference_model
from src.api.schemas import RecommendationItem, RecommendRequest, RecommendResponse

app = FastAPI(
    title="E-Commerce Recommender API",
    version="1.0.0",
)

inference_model = load_inference_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    try:
        recs_df = inference_model.recommend(
            user_id=request.user_id,
            n_candidates=request.n_candidates,
            top_k=request.top_k,
            filter_already_liked_items=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    recommendations = [
        RecommendationItem(
            item_id=int(row["itemid"]),
            score=float(row["rerank_score"]),
            als_score=(
                float(row["als_score"]) if row.get("als_score") is not None else None
            ),
            categoryid=(
                int(row["categoryid"]) if row.get("categoryid") is not None else None
            ),
        )
        for row in recs_df.to_dicts()
    ]

    return RecommendResponse(
        user_id=request.user_id,
        recommendations=recommendations,
    )
