from __future__ import annotations

from typing import Any


def comparison_rows(suite: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, payload in suite.items():
        m = payload["metrics"]
        rows.append(
            {
                "model": name,
                "backend": m.get("backend"),
                "n": m.get("n"),
                "root_cause_accuracy": m["root_cause"]["accuracy"],
                "root_cause_macro_f1": m["root_cause"]["macro_f1"],
                "severity_accuracy": m["severity"]["accuracy"],
                "severity_macro_f1": m["severity"]["macro_f1"],
                "json_parse_success": m["json_parse_success"],
                "pydantic_schema_success": m["pydantic_schema_success"],
                "supported_evidence_rate": m["supported_evidence_rate"],
                "unsupported_evidence_rate": m["unsupported_evidence_rate"],
                "action_f1": m["action_f1"],
                "latency_p50_ms": m["latency"]["p50_ms"],
                "latency_p95_ms": m["latency"]["p95_ms"],
            }
        )
    return rows
