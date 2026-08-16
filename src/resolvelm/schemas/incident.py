from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from resolvelm.schemas.analysis import RecommendedAction, Severity


class LabelOrigin(StrEnum):
    ORIGINAL = "original"
    DERIVED = "derived"
    DETERMINISTIC_TEMPLATE = "deterministic_template"
    SYNTHETIC_PREFERENCE = "synthetic_preference"


class LogEvent(BaseModel):
    event_id: str = Field(min_length=1)
    timestamp: datetime
    level: str
    component: str
    message: str = Field(min_length=1, max_length=2000)


class MetricEvent(BaseModel):
    event_id: str = Field(min_length=1)
    timestamp: datetime
    name: str
    value: float
    unit: str = "count"


class TraceEvent(BaseModel):
    event_id: str = Field(min_length=1)
    timestamp: datetime
    span: str
    duration_ms: float = Field(ge=0)
    status: str


class IncidentRecord(BaseModel):
    incident_id: str = Field(min_length=3)
    family_id: str = Field(min_length=3)
    source_dataset: str
    split: str = Field(pattern=r"^(train|val|test|unsplit)$")
    timestamp: datetime
    service: str
    component: str
    environment: str
    symptoms: list[str] = Field(min_length=1, max_length=20)
    log_events: list[LogEvent] = Field(min_length=1)
    metric_events: list[MetricEvent] = Field(default_factory=list)
    trace_events: list[TraceEvent] = Field(default_factory=list)
    root_cause_code: str
    root_cause_text: str
    severity: Severity
    evidence_ids: list[str] = Field(min_length=1)
    action_ids: list[str] = Field(min_length=1)
    recommended_actions: list[RecommendedAction] = Field(min_length=1)
    label_origin: LabelOrigin
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("symptoms")
    @classmethod
    def symptoms_nonempty(cls, values: list[str]) -> list[str]:
        cleaned = [s.strip() for s in values if s and s.strip()]
        if not cleaned:
            raise ValueError("symptoms must contain non-empty strings")
        return cleaned

    @model_validator(mode="after")
    def evidence_in_context(self) -> IncidentRecord:
        known = {e.event_id for e in self.log_events}
        known.update(e.event_id for e in self.metric_events)
        known.update(e.event_id for e in self.trace_events)
        missing = [eid for eid in self.evidence_ids if eid not in known]
        if missing:
            raise ValueError(f"evidence_ids not in incident context: {missing}")
        return self
