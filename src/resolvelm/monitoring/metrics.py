from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS = Counter(
    "resolvelm_requests_total",
    "API requests",
    ["endpoint", "status"],
)
LATENCY = Histogram(
    "resolvelm_request_latency_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
PARSE_FAILURES = Counter(
    "resolvelm_parse_failures_total",
    "Invalid model JSON after retry",
)
PAYLOAD_REJECTS = Counter(
    "resolvelm_payload_rejects_total",
    "Rejected oversized or invalid payloads",
    ["reason"],
)


def metrics_output() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
