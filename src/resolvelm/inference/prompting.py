from __future__ import annotations

from resolvelm.data.generator import SYSTEM_PROMPT, render_user_prompt
from resolvelm.schemas.api import AnalyzeRequest

__all__ = ["SYSTEM_PROMPT", "request_to_prompt"]


def request_to_prompt(request: AnalyzeRequest) -> str:
    payload = {
        "service": request.service,
        "environment": request.environment,
        "symptoms": request.symptoms,
        "log_events": [e.model_dump(mode="json") for e in request.log_events],
        "metric_events": [e.model_dump(mode="json") for e in request.metric_events],
        "trace_events": [e.model_dump(mode="json") for e in request.trace_events],
    }
    user = render_user_prompt(payload)
    return f"System: {SYSTEM_PROMPT}\nUser: {user}"
