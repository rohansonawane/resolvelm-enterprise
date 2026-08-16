from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any

from pydantic import ValidationError

from resolvelm.schemas.incident import IncidentRecord

VALID_SEVERITIES = {"SEV-1", "SEV-2", "SEV-3", "SEV-4"}
VALID_ORIGINS = {"original", "derived", "deterministic_template", "synthetic_preference"}


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _body_hash(record: IncidentRecord) -> str:
    blob = " ".join(record.symptoms) + " " + " ".join(e.message for e in record.log_events)
    return hashlib.sha256(_norm_text(blob).encode("utf-8")).hexdigest()


def _ngrams(text: str, n: int) -> set[str]:
    tokens = _norm_text(text).split()
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def validate_incidents(
    records: list[IncidentRecord],
    *,
    near_dup_ngram: int = 8,
    near_dup_jaccard: float = 0.92,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    hashes: dict[str, str] = {}
    encodings = 0
    long_bodies = 0

    for rec in records:
        try:
            IncidentRecord.model_validate(rec.model_dump())
        except ValidationError as exc:
            issues.append({"type": "schema", "id": rec.incident_id, "detail": str(exc)})
            continue
        if rec.incident_id in seen_ids:
            issues.append({"type": "duplicate_id", "id": rec.incident_id})
        seen_ids.add(rec.incident_id)
        if rec.severity.value not in VALID_SEVERITIES:
            issues.append(
                {"type": "invalid_severity", "id": rec.incident_id, "value": rec.severity}
            )
        if rec.label_origin.value not in VALID_ORIGINS:
            issues.append({"type": "invalid_label_origin", "id": rec.incident_id})
        if not rec.log_events:
            issues.append({"type": "empty_body", "id": rec.incident_id})
        body = " ".join(e.message for e in rec.log_events)
        if any(ord(ch) == 0 for ch in body):
            encodings += 1
            issues.append({"type": "encoding", "id": rec.incident_id})
        if len(body) > 8000:
            long_bodies += 1
            issues.append({"type": "extreme_length", "id": rec.incident_id, "chars": len(body)})
        digest = _body_hash(rec)
        if digest in hashes:
            issues.append(
                {
                    "type": "exact_duplicate_body",
                    "id": rec.incident_id,
                    "other": hashes[digest],
                }
            )
        else:
            hashes[digest] = rec.incident_id

    ngram_index: list[tuple[str, str, set[str]]] = []
    for rec in records:
        text = " ".join(e.message for e in rec.log_events)
        ngram_index.append((rec.incident_id, rec.split, _ngrams(text, near_dup_ngram)))

    near_dups = 0
    for i, (id_a, split_a, grams_a) in enumerate(ngram_index):
        for id_b, split_b, grams_b in ngram_index[i + 1 : i + 12]:
            if not grams_a or not grams_b:
                continue
            jacc = len(grams_a & grams_b) / max(1, len(grams_a | grams_b))
            if jacc >= near_dup_jaccard and split_a != split_b and split_a != "unsplit":
                near_dups += 1
                issues.append(
                    {
                        "type": "near_duplicate_across_split",
                        "id": id_a,
                        "other": id_b,
                        "jaccard": round(jacc, 3),
                    }
                )

    by_split = Counter(r.split for r in records)
    by_origin = Counter(r.label_origin.value for r in records)
    by_code = Counter(r.root_cause_code for r in records)
    families: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        families[rec.family_id].add(rec.split)
    leaked_families = {
        fid: sorted(splits) for fid, splits in families.items() if len(splits - {"unsplit"}) > 1
    }

    report = {
        "n_records": len(records),
        "n_issues": len(issues),
        "issue_types": dict(Counter(i["type"] for i in issues)),
        "issues_preview": issues[:50],
        "split_counts": dict(by_split),
        "label_origin_counts": dict(by_origin),
        "root_cause_counts": dict(by_code),
        "encoding_problems": encodings,
        "extreme_length_count": long_bodies,
        "near_duplicate_across_split": near_dups,
        "family_split_leakage": leaked_families,
        "passed": len(issues) == 0 and not leaked_families,
    }
    return report
