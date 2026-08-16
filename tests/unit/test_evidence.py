from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from resolvelm.schemas.incident import IncidentRecord, LabelOrigin, LogEvent, Severity


def _base(**kwargs):
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    payload = dict(
        incident_id="INC-X",
        family_id="FAM-X",
        source_dataset="hdfs",
        split="train",
        timestamp=ts,
        service="hdfs",
        component="dn",
        environment="prod",
        symptoms=["disk filling"],
        log_events=[
            LogEvent(
                event_id="log-1",
                timestamp=ts,
                level="ERROR",
                component="dn",
                message="No space left on device",
            )
        ],
        root_cause_code="DISK_FULL",
        root_cause_text="disk full",
        severity=Severity.SEV_2,
        evidence_ids=["log-1"],
        action_ids=["expand_volume"],
        recommended_actions=[
            {"action_id": "expand_volume", "priority": 1, "description": "expand volume"}
        ],
        label_origin=LabelOrigin.DERIVED,
    )
    payload.update(kwargs)
    return IncidentRecord(**payload)


def test_valid_evidence_passes() -> None:
    rec = _base()
    assert rec.evidence_ids == ["log-1"]


def test_outside_context_evidence_fails() -> None:
    with pytest.raises(ValidationError):
        _base(evidence_ids=["metric-9"])
