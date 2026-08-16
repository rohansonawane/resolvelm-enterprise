"""Parse model text into IncidentAnalysis. Retry/repair once; never invent a valid result."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from resolvelm.schemas.analysis import IncidentAnalysis

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class ParseResult:
    def __init__(
        self,
        analysis: IncidentAnalysis | None,
        raw: str,
        attempts: int,
        error: str | None = None,
    ) -> None:
        self.analysis = analysis
        self.raw = raw
        self.attempts = attempts
        self.error = error

    @property
    def ok(self) -> bool:
        return self.analysis is not None


def _extract_object(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    match = JSON_BLOCK.search(stripped)
    return match.group(0) if match else None


def _light_repair(text: str) -> str:
    repaired = text.strip()
    repaired = repaired.replace("'", '"')
    repaired = re.sub(r",\s*}", "}", repaired)
    repaired = re.sub(r",\s*]", "]", repaired)
    return repaired


def parse_analysis_with_retry(text: str) -> ParseResult:
    attempts = 1
    blob = _extract_object(text)
    if blob:
        try:
            payload: Any = json.loads(blob)
            return ParseResult(IncidentAnalysis.model_validate(payload), text, attempts)
        except (json.JSONDecodeError, ValidationError) as exc:
            attempts = 2
            try:
                payload = json.loads(_light_repair(blob))
                return ParseResult(IncidentAnalysis.model_validate(payload), text, attempts)
            except (json.JSONDecodeError, ValidationError):
                return ParseResult(None, text, attempts, error=str(exc))
    attempts = 2
    repaired = _extract_object(_light_repair(text))
    if repaired:
        try:
            payload = json.loads(repaired)
            return ParseResult(IncidentAnalysis.model_validate(payload), text, attempts)
        except (json.JSONDecodeError, ValidationError) as exc:
            return ParseResult(None, text, attempts, error=str(exc))
    return ParseResult(None, text, attempts, error="no JSON object found in model output")
