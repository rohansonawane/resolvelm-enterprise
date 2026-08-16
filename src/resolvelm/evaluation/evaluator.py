from __future__ import annotations

import time
from typing import Any

from resolvelm.data.convert import incident_payload
from resolvelm.data.generator import render_user_prompt
from resolvelm.evaluation.metrics import classification_report_safe, latency_stats, set_prf
from resolvelm.inference.prompting import SYSTEM_PROMPT
from resolvelm.inference.protocol import ModelBackend
from resolvelm.inference.repair import parse_analysis_with_retry
from resolvelm.schemas.incident import IncidentRecord


def _prompt_for(record: IncidentRecord) -> str:
    return f"System: {SYSTEM_PROMPT}\nUser: {render_user_prompt(incident_payload(record))}"


def evaluate_backend(
    backend: ModelBackend,
    records: list[IncidentRecord],
    *,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    del timeout_s
    predictions: list[dict[str, Any]] = []
    y_cause_true: list[str] = []
    y_cause_pred: list[str] = []
    y_sev_true: list[str] = []
    y_sev_pred: list[str] = []
    json_ok = 0
    schema_ok = 0
    supported = 0
    unsupported = 0
    evidence_considered = 0
    action_scores: list[dict[str, float]] = []
    latencies: list[float] = []

    for rec in records:
        prompt = _prompt_for(rec)
        known_ids = {e.event_id for e in rec.log_events + rec.metric_events + rec.trace_events}
        t0 = time.perf_counter()
        raw = backend.generate_sync(prompt)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        parsed = parse_analysis_with_retry(raw)
        row: dict[str, Any] = {
            "incident_id": rec.incident_id,
            "family_id": rec.family_id,
            "gold_root_cause_code": rec.root_cause_code,
            "gold_severity": rec.severity.value,
            "raw_preview": raw[:500],
            "parse_attempts": parsed.attempts,
            "latency_ms": latency,
            "json_ok": parsed.analysis is not None
            or parsed.error != "no JSON object found in model output",
            "schema_ok": parsed.ok,
            "error": parsed.error,
        }
        # json parse success: extracted JSON object even if schema failed
        if parsed.analysis is not None:
            json_ok += 1
            schema_ok += 1
            analysis = parsed.analysis
            y_cause_true.append(rec.root_cause_code)
            y_cause_pred.append(analysis.root_cause_code)
            y_sev_true.append(rec.severity.value)
            y_sev_pred.append(analysis.severity.value)
            for ev in analysis.evidence:
                evidence_considered += 1
                if ev.source_id in known_ids:
                    supported += 1
                else:
                    unsupported += 1
            pred_actions = {a.action_id for a in analysis.recommended_actions}
            gold_actions = set(rec.action_ids)
            action_scores.append(set_prf(pred_actions, gold_actions))
            row.update(
                {
                    "pred_root_cause_code": analysis.root_cause_code,
                    "pred_severity": analysis.severity.value,
                    "confidence": analysis.confidence,
                    "needs_human_review": analysis.needs_human_review,
                }
            )
        else:
            # If JSON existed but schema failed, count json separately
            if (
                parsed.error
                and "JSON" not in (parsed.error or "")
                and "no JSON" not in (parsed.error or "")
            ):
                json_ok += 1
            y_cause_true.append(rec.root_cause_code)
            y_cause_pred.append("__INVALID__")
            y_sev_true.append(rec.severity.value)
            y_sev_pred.append("__INVALID__")
            row["pred_root_cause_code"] = None
            row["pred_severity"] = None
        predictions.append(row)

    n = max(len(records), 1)
    action_f1 = float(sum(s["f1"] for s in action_scores) / max(len(action_scores), 1))
    action_p = float(sum(s["precision"] for s in action_scores) / max(len(action_scores), 1))
    action_r = float(sum(s["recall"] for s in action_scores) / max(len(action_scores), 1))
    metrics = {
        "backend": backend.name,
        "model_version": backend.version,
        "n": len(records),
        "root_cause": classification_report_safe(y_cause_true, y_cause_pred),
        "severity": classification_report_safe(y_sev_true, y_sev_pred),
        "json_parse_success": json_ok / n,
        "pydantic_schema_success": schema_ok / n,
        "supported_evidence_rate": supported / max(evidence_considered, 1),
        "unsupported_evidence_rate": unsupported / max(evidence_considered, 1),
        "action_precision": action_p,
        "action_recall": action_r,
        "action_f1": action_f1,
        "latency": latency_stats(latencies),
        "tokens_per_sec": None,
        "peak_memory_mb": None,
        "tokens_per_sec_note": "Not benchmarked on this hardware.",
        "peak_memory_note": "Not benchmarked on this hardware.",
    }
    return {"metrics": metrics, "predictions": predictions}


def evaluate_suite(
    backends: dict[str, ModelBackend],
    records: list[IncidentRecord],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for key, backend in backends.items():
        results[key] = evaluate_backend(backend, records)
    return results
