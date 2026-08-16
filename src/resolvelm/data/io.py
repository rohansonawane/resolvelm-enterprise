from __future__ import annotations

from typing import Any

from resolvelm.data.generator import gold_analysis, render_user_prompt
from resolvelm.schemas.incident import IncidentRecord


def record_to_analyze_payload(record: IncidentRecord) -> dict[str, Any]:
    return {
        "service": record.service,
        "environment": record.environment,
        "symptoms": record.symptoms,
        "log_events": [e.model_dump(mode="json") for e in record.log_events],
        "metric_events": [e.model_dump(mode="json") for e in record.metric_events],
        "trace_events": [e.model_dump(mode="json") for e in record.trace_events],
    }


def gold_json(record: IncidentRecord) -> dict[str, Any]:
    return gold_analysis(record)


def user_text(record: IncidentRecord) -> str:
    return render_user_prompt(record_to_analyze_payload(record))
