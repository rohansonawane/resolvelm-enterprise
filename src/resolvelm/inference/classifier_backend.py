"""Bag-of-words classifier backend used when HF weights are unavailable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from resolvelm.data.generator import ROOT_CAUSES
from resolvelm.inference.rule_backend import _context_ids
from resolvelm.schemas.analysis import IncidentAnalysis


class ClassifierBackend:
    name = "classifier"
    version = "classifier-unloaded"

    def __init__(self, path: str | Path) -> None:
        payload: dict[str, Any] = joblib.load(path)
        self.model = payload["model"]
        self.vectorizer = payload["vectorizer"]
        self.severity_model = payload.get("severity_model")
        self.version = payload.get("version", Path(path).as_posix())
        self.name = payload.get("name", "classifier")

    def generate_sync(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        del max_new_tokens
        features = self.vectorizer.transform([prompt])
        code = str(self.model.predict(features)[0])
        meta = ROOT_CAUSES.get(code) or ROOT_CAUSES["SERVICE_TIMEOUT"]
        if code not in ROOT_CAUSES:
            code = "SERVICE_TIMEOUT"
            meta = ROOT_CAUSES[code]
        severity = meta["severity"].value
        if self.severity_model is not None:
            severity = str(self.severity_model.predict(features)[0])
        source_ids = _context_ids(prompt) or ["log-1"]
        payload = {
            "severity": severity,
            "root_cause_code": code,
            "root_cause": meta["text"],
            "confidence": 0.67,
            "evidence": [
                {"source_id": source_ids[0], "reason": "Classifier decision using incident text"}
            ],
            "recommended_actions": [
                {
                    "action_id": aid,
                    "priority": i,
                    "description": desc,
                }
                for i, (aid, desc) in enumerate(meta["actions"], start=1)
            ],
            "needs_human_review": True,
        }
        IncidentAnalysis.model_validate(payload)
        return json.dumps(payload, separators=(",", ":"))

    async def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        return self.generate_sync(prompt, max_new_tokens=max_new_tokens)
