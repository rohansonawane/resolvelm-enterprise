from __future__ import annotations

from typing import Any


def collect_failures(predictions: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for row in predictions:
        if row.get("pred_root_cause_code") != row.get("gold_root_cause_code") or not row.get(
            "schema_ok"
        ):
            failed.append(
                {
                    "incident_id": row["incident_id"],
                    "gold": row.get("gold_root_cause_code"),
                    "pred": row.get("pred_root_cause_code"),
                    "gold_severity": row.get("gold_severity"),
                    "pred_severity": row.get("pred_severity"),
                    "error": row.get("error"),
                    "raw_preview": row.get("raw_preview"),
                }
            )
        if len(failed) >= limit:
            break
    return failed
