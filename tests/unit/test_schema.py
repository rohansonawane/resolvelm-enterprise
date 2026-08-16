from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from resolvelm.data.generator import generate_corpus, gold_analysis
from resolvelm.schemas.analysis import IncidentAnalysis
from resolvelm.schemas.incident import IncidentRecord, LabelOrigin, LogEvent, Severity


def test_incident_analysis_accepts_contract() -> None:
    payload = {
        "severity": "SEV-2",
        "root_cause_code": "DB_CONNECTION_POOL_EXHAUSTION",
        "root_cause": "Database connection pool exhaustion",
        "confidence": 0.91,
        "evidence": [{"source_id": "log-17", "reason": "Connection pool reported exhaustion"}],
        "recommended_actions": [
            {
                "action_id": "inspect_connection_leaks",
                "priority": 1,
                "description": "Inspect ORM sessions and connection leaks",
            }
        ],
        "needs_human_review": False,
    }
    parsed = IncidentAnalysis.model_validate(payload)
    assert parsed.root_cause_code == "DB_CONNECTION_POOL_EXHAUSTION"


def test_incident_analysis_rejects_placeholder_code() -> None:
    with pytest.raises(ValidationError):
        IncidentAnalysis.model_validate(
            {
                "severity": "SEV-2",
                "root_cause_code": "UNKNOWN",
                "root_cause": "not sure",
                "confidence": 0.1,
                "evidence": [{"source_id": "log-1", "reason": "guess"}],
                "recommended_actions": [
                    {"action_id": "look", "priority": 1, "description": "look around"}
                ],
                "needs_human_review": True,
            }
        )


def test_evidence_must_exist_in_context() -> None:
    ts = datetime.now(UTC)
    with pytest.raises(ValidationError):
        IncidentRecord(
            incident_id="INC-1",
            family_id="FAM-1",
            source_dataset="hdfs",
            split="train",
            timestamp=ts,
            service="hdfs",
            component="dn",
            environment="prod",
            symptoms=["disk"],
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
            evidence_ids=["log-99"],
            action_ids=["expand_volume"],
            recommended_actions=[
                {"action_id": "expand_volume", "priority": 1, "description": "add disk"}
            ],
            label_origin=LabelOrigin.DERIVED,
        )


def test_generated_records_validate() -> None:
    records = generate_corpus(4, 2)
    assert len(records) == 8
    for rec in records:
        gold = gold_analysis(rec)
        IncidentAnalysis.model_validate(gold)
