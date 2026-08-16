from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Severity(StrEnum):
    SEV_1 = "SEV-1"
    SEV_2 = "SEV-2"
    SEV_3 = "SEV-3"
    SEV_4 = "SEV-4"


class EvidenceItem(BaseModel):
    source_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=3, max_length=500)


class RecommendedAction(BaseModel):
    action_id: str = Field(min_length=1, max_length=128)
    priority: int = Field(ge=1, le=10)
    description: str = Field(min_length=3, max_length=800)


class IncidentAnalysis(BaseModel):
    """Strict JSON contract returned to operators. Invalid model text must not be coerced."""

    severity: Severity
    root_cause_code: str = Field(min_length=3, max_length=80, pattern=r"^[A-Z][A-Z0-9_]+$")
    root_cause: str = Field(min_length=5, max_length=400)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=20)
    recommended_actions: list[RecommendedAction] = Field(min_length=1, max_length=12)
    needs_human_review: bool

    @field_validator("root_cause_code")
    @classmethod
    def code_not_placeholder(cls, value: str) -> str:
        if value in {"UNKNOWN", "N_A", "NONE", "TODO"}:
            raise ValueError("root_cause_code must be a concrete taxonomy code")
        return value


def analysis_json_schema() -> dict[str, Any]:
    return IncidentAnalysis.model_json_schema()
