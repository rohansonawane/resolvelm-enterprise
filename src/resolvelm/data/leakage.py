from __future__ import annotations

from resolvelm.schemas.incident import IncidentRecord


def exact_overlap_ids(train: list[IncidentRecord], test: list[IncidentRecord]) -> list[str]:
    train_ids = {r.incident_id for r in train}
    return sorted(r.incident_id for r in test if r.incident_id in train_ids)


def family_leakage(train: list[IncidentRecord], test: list[IncidentRecord]) -> list[str]:
    train_f = {r.family_id for r in train}
    return sorted({r.family_id for r in test if r.family_id in train_f})
