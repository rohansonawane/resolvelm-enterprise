from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from resolvelm.schemas.analysis import IncidentAnalysis
from resolvelm.schemas.incident import LogEvent, MetricEvent, TraceEvent


class AnalyzeRequest(BaseModel):
    service: str = Field(min_length=1, max_length=128)
    environment: str = Field(min_length=1, max_length=64)
    symptoms: list[str] = Field(min_length=1, max_length=20)
    log_events: list[LogEvent] = Field(default_factory=list, max_length=80)
    metric_events: list[MetricEvent] = Field(default_factory=list, max_length=40)
    trace_events: list[TraceEvent] = Field(default_factory=list, max_length=40)
    demo_id: str | None = None

    @field_validator("symptoms")
    @classmethod
    def strip_symptoms(cls, values: list[str]) -> list[str]:
        cleaned = [s.strip() for s in values if s and s.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty symptom is required")
        return cleaned


class AnalysisErrorPayload(BaseModel):
    error: Literal["invalid_model_output", "backend_unavailable", "timeout", "validation_error"]
    message: str
    parse_attempts: int = 0
    raw_output_preview: str | None = None


class AnalyzeResponse(BaseModel):
    request_id: str
    model_version: str
    backend: str
    latency_ms: float
    analysis: IncidentAnalysis | None = None
    error: AnalysisErrorPayload | None = None
    needs_human_review: bool = True


class BatchAnalyzeRequest(BaseModel):
    incidents: list[AnalyzeRequest] = Field(min_length=1, max_length=32)


class BatchAnalyzeResponse(BaseModel):
    request_id: str
    results: list[AnalyzeResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    backend: str
    device: str


class ModelInfo(BaseModel):
    name: str
    version: str
    backend: str
    production_target: str
    executed_locally: bool


class BenchmarkRow(BaseModel):
    model: str
    metrics: dict[str, Any]
