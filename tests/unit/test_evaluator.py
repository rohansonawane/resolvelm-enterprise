from resolvelm.data.generator import generate_corpus
from resolvelm.data.split import grouped_temporal_split
from resolvelm.evaluation.evaluator import evaluate_backend
from resolvelm.inference.repair import parse_analysis_with_retry
from resolvelm.inference.rule_backend import RuleBasedBackend


def test_rule_backend_evaluates() -> None:
    records = [r for r in grouped_temporal_split(generate_corpus(12, 1)) if r.split == "test"]
    result = evaluate_backend(RuleBasedBackend(), records)
    metrics = result["metrics"]
    assert metrics["n"] == len(records)
    assert 0 <= metrics["root_cause"]["accuracy"] <= 1
    assert metrics["pydantic_schema_success"] == 1.0
    assert metrics["json_parse_success"] == 1.0
    assert metrics["tokens_per_sec"] is None


def test_repair_rejects_garbage() -> None:
    parsed = parse_analysis_with_retry("I think the disk is full maybe")
    assert parsed.analysis is None
    assert parsed.attempts == 2


def test_repair_accepts_fenced_json() -> None:
    blob = """```json
{"severity":"SEV-2","root_cause_code":"DISK_FULL","root_cause":"Local disk is full",
 "confidence":0.8,"evidence":[{"source_id":"log-1","reason":"no space"}],
 "recommended_actions":[{"action_id":"expand_volume","priority":1,"description":"add disk"}],
 "needs_human_review":true}
```"""
    parsed = parse_analysis_with_retry(blob)
    assert parsed.ok
    assert parsed.analysis is not None
    assert parsed.analysis.root_cause_code == "DISK_FULL"
