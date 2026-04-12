from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)

HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)

RECOMMEND_REQUESTS_TOTAL = Counter(
    "recommend_requests_total",
    "Total number of recommendation requests",
)

RECOMMEND_REQUEST_LATENCY_SECONDS = Histogram(
    "recommend_request_latency_seconds",
    "Latency of recommendation generation in seconds",
)

RECOMMENDATIONS_RETURNED_TOTAL = Counter(
    "recommendations_returned_total",
    "Total number of returned recommendations",
)

INFERENCE_ERRORS_TOTAL = Counter(
    "inference_errors_total",
    "Total number of inference errors",
)

COLD_USERS_TOTAL = Counter(
    "cold_users_total",
    "Total number of cold-start users in API requests",
)

RECOMMENDATION_SCORE_MEAN = Gauge(
    "recommendation_score_mean",
    "Mean rerank score of returned recommendations",
)
