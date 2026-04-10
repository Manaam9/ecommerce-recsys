import random

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="E-commerce Recommender")


# Request schema
class RecommendRequest(BaseModel):
    visitorid: int
    k: int = 10


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Recommendation endpoint
@app.post("/recommend")
def recommend(req: RecommendRequest):
    # TODO: заменить на реальную модель
    mock_items = list(range(1000, 1100))
    recs = random.sample(mock_items, req.k)

    return {"visitorid": req.visitorid, "recommendations": recs}
