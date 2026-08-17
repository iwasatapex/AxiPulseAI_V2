from prometheus_client import (
    Counter,
    Histogram,
    Gauge
)


REQUEST_COUNT = Counter(
    "axipulse_api_requests_total",
    "Total API requests",
    [
        "method",
        "endpoint",
        "status"
    ]
)


REQUEST_LATENCY = Histogram(
    "axipulse_api_request_latency_seconds",
    "API request latency",
    [
        "endpoint"
    ]
)


ACTIVE_REQUESTS = Gauge(
    "axipulse_active_requests",
    "Active API requests"
)


ENGINE_CALLS = Counter(
    "axipulse_engine_calls_total",
    "AI engine executions",
    [
        "engine"
    ]
)


def record_engine_call(engine):

    ENGINE_CALLS.labels(
        engine=engine
    ).inc()
