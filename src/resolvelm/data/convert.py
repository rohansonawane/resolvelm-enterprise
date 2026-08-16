"""Convert incidents into SFT chat examples and DPO preference pairs.

Tokenizer chat templates are applied only when a Hugging Face tokenizer is provided.
Special tokens are never invented.
"""

from __future__ import annotations

import json
from typing import Any

from resolvelm.data.generator import (
    REJECTION_CATEGORIES,
    SYSTEM_PROMPT,
    gold_analysis,
    render_user_prompt,
)
from resolvelm.schemas.incident import IncidentRecord

WRONG_CODES = {
    "DISK_FULL": "NETWORK_PARTITION",
    "NETWORK_PARTITION": "DISK_FULL",
    "NODE_FAILURE": "SERVICE_TIMEOUT",
    "DB_CONNECTION_POOL_EXHAUSTION": "AUTH_TOKEN_EXPIRED",
    "SERVICE_TIMEOUT": "DB_CONNECTION_POOL_EXHAUSTION",
    "HDFS_BLOCK_MISSING": "PERMISSION_DENIED",
    "YARN_CONTAINER_OOM": "CONFIG_MISMATCH",
    "OPENSTACK_NOVA_SCHEDULER_FAILURE": "NODE_FAILURE",
    "MEMORY_PRESSURE": "DISK_FULL",
    "PERMISSION_DENIED": "HDFS_BLOCK_MISSING",
    "CONFIG_MISMATCH": "YARN_CONTAINER_OOM",
    "AUTH_TOKEN_EXPIRED": "SERVICE_TIMEOUT",
}


def incident_payload(record: IncidentRecord) -> dict[str, Any]:
    return {
        "service": record.service,
        "environment": record.environment,
        "symptoms": record.symptoms,
        "log_events": [e.model_dump(mode="json") for e in record.log_events],
        "metric_events": [e.model_dump(mode="json") for e in record.metric_events],
        "trace_events": [e.model_dump(mode="json") for e in record.trace_events],
    }


def apply_chat_template(messages: list[dict[str, str]], tokenizer: Any | None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    parts: list[str] = []
    for msg in messages:
        parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
    return "\n".join(parts)


def incidents_to_sft(
    records: list[IncidentRecord], tokenizer: Any | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        user = render_user_prompt(incident_payload(rec))
        assistant = json.dumps(gold_analysis(rec), separators=(",", ":"))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        rows.append(
            {
                "incident_id": rec.incident_id,
                "split": rec.split,
                "messages": messages,
                "text": apply_chat_template(messages, tokenizer),
                "root_cause_code": rec.root_cause_code,
                "severity": rec.severity.value,
            }
        )
    return rows


def _corrupt(record: IncidentRecord, category: str, chosen: dict[str, Any]) -> str:
    rejected = json.loads(json.dumps(chosen))
    if category == "wrong_root_cause":
        rejected["root_cause_code"] = WRONG_CODES.get(record.root_cause_code, "NETWORK_PARTITION")
        rejected["root_cause"] = "Misattributed failure"
    elif category == "unsupported_evidence":
        rejected["evidence"] = [{"source_id": "log-999", "reason": "Invented log line"}]
    elif category == "invalid_json":
        return "{severity: SEV-2, not json"
    elif category == "missing_evidence":
        rejected["evidence"] = []
    elif category == "dangerous_remediation":
        rejected["recommended_actions"] = [
            {
                "action_id": "drop_production_db",
                "priority": 1,
                "description": "Drop the production database to clear connections",
            }
        ]
    elif category == "vague_response":
        rejected["root_cause"] = "Something is wrong"
        rejected["root_cause_code"] = "ISSUE_GENERIC"
        rejected["confidence"] = 0.2
    elif category == "severity_error":
        rejected["severity"] = "SEV-4" if record.severity.value == "SEV-1" else "SEV-1"
    elif category == "excessive_certainty":
        rejected["confidence"] = 0.99
        rejected["needs_human_review"] = False
        rejected["evidence"] = [{"source_id": "log-1", "reason": "Guessed without corroboration"}]
    return json.dumps(rejected, separators=(",", ":"))


def incidents_to_dpo(
    records: list[IncidentRecord],
    *,
    pairs_per_record: int = 2,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    """Build preference pairs from TRAIN/VAL only. Callers must not pass test records."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        if rec.split == "test":
            continue
        chosen = gold_analysis(rec)
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(incident_payload(rec))},
        ]
        prompt = apply_chat_template(prompt_messages, tokenizer)
        for i, _category in enumerate(REJECTION_CATEGORIES):
            if i >= pairs_per_record:
                break
            # Rotate category by incident index so the taxonomy is covered.
            cat = REJECTION_CATEGORIES[(hash(rec.incident_id) + i) % len(REJECTION_CATEGORIES)]
            rows.append(
                {
                    "incident_id": rec.incident_id,
                    "split": rec.split,
                    "prompt": prompt,
                    "chosen": json.dumps(chosen, separators=(",", ":")),
                    "rejected": _corrupt(rec, cat, chosen),
                    "rejection_category": cat,
                    "label_origin": "synthetic_preference",
                    "root_cause_code": rec.root_cause_code,
                }
            )
    return rows
