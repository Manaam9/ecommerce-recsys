import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.dependencies import load_inference_model
from src.api.metrics import (
    COLD_USERS_TOTAL,
    HTTP_REQUEST_LATENCY_SECONDS,
    HTTP_REQUESTS_TOTAL,
    INFERENCE_ERRORS_TOTAL,
    RECOMMEND_REQUEST_LATENCY_SECONDS,
    RECOMMEND_REQUESTS_TOTAL,
    RECOMMENDATION_SCORE_MEAN,
    RECOMMENDATIONS_RETURNED_TOTAL,
)
from src.api.schemas import RecommendationItem, RecommendRequest, RecommendResponse

app = FastAPI(
    title="E-Commerce Recommender API",
    version="1.0.0",
)

inference_model = load_inference_model()


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = None
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start_time
        endpoint = request.url.path
        method = request.method

        HTTP_REQUESTS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code),
        ).inc()

        HTTP_REQUEST_LATENCY_SECONDS.labels(
            method=method,
            endpoint=endpoint,
        ).observe(elapsed)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    RECOMMEND_REQUESTS_TOTAL.inc()
    start_time = time.perf_counter()

    try:
        if request.user_id not in inference_model.user2idx:
            COLD_USERS_TOTAL.inc()

        recs_df = inference_model.recommend(
            user_id=request.user_id,
            n_candidates=request.n_candidates,
            top_k=request.top_k,
            filter_already_liked_items=True,
        )
    except Exception as e:
        INFERENCE_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        elapsed = time.perf_counter() - start_time
        RECOMMEND_REQUEST_LATENCY_SECONDS.observe(elapsed)

    recommendations = []
    scores = []

    for row in recs_df.to_dicts():
        score = float(row["rerank_score"])
        scores.append(score)

        recommendations.append(
            RecommendationItem(
                item_id=int(row["itemid"]),
                score=score,
                als_score=(
                    float(row["als_score"])
                    if row.get("als_score") is not None
                    else None
                ),
                categoryid=(
                    int(row["categoryid"])
                    if row.get("categoryid") is not None
                    else None
                ),
            )
        )

    RECOMMENDATIONS_RETURNED_TOTAL.inc(len(recommendations))

    if scores:
        RECOMMENDATION_SCORE_MEAN.set(sum(scores) / len(scores))

    return RecommendResponse(
        user_id=request.user_id,
        recommendations=recommendations,
    )
