from __future__ import annotations

import uuid
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from resolvelm.api.settings import get_settings
from resolvelm.monitoring.metrics import LATENCY, PAYLOAD_REJECTS, REQUESTS


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        settings = get_settings()
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_payload_bytes:
            PAYLOAD_REJECTS.labels(reason="oversized").inc()
            REQUESTS.labels(endpoint=request.url.path, status="413").inc()
            return JSONResponse(
                {
                    "error": "payload_too_large",
                    "message": f"Payload exceeds {settings.max_payload_bytes} bytes",
                    "request_id": request_id,
                },
                status_code=413,
                headers={"x-request-id": request_id},
            )
        with LATENCY.labels(endpoint=request.url.path).time():
            response = await call_next(request)
        response.headers["x-request-id"] = request_id
        REQUESTS.labels(endpoint=request.url.path, status=str(response.status_code)).inc()
        return response
