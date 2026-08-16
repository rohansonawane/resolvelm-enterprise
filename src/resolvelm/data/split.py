from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime

from resolvelm.schemas.incident import IncidentRecord


def _body_key(record: IncidentRecord) -> str:
    blob = "||".join(record.symptoms + [e.message for e in record.log_events])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def dedup_records(records: list[IncidentRecord]) -> list[IncidentRecord]:
    seen: set[str] = set()
    unique: list[IncidentRecord] = []
    for rec in records:
        key = _body_key(rec)
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    return unique


def grouped_temporal_split(
    records: list[IncidentRecord],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    group_key: str = "family_id",
    time_key: str = "timestamp",
) -> list[IncidentRecord]:
    """Keep all variants of a family in one split. Sort families by earliest timestamp."""
    records = dedup_records(records)
    families: dict[str, list[IncidentRecord]] = defaultdict(list)
    for rec in records:
        families[getattr(rec, group_key)].append(rec)

    def earliest(items: list[IncidentRecord]) -> datetime:
        return min(getattr(item, time_key) for item in items)

    ordered = sorted(families.items(), key=lambda kv: earliest(kv[1]))
    n = len(ordered)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    assignments: dict[str, str] = {}
    for i, (fid, _) in enumerate(ordered):
        if i < n_train:
            assignments[fid] = "train"
        elif i < n_train + n_val:
            assignments[fid] = "val"
        else:
            assignments[fid] = "test"

    split_records: list[IncidentRecord] = []
    for rec in records:
        split_records.append(rec.model_copy(update={"split": assignments[rec.family_id]}))
    return split_records
