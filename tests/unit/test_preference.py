import json

from resolvelm.data.convert import _corrupt, incidents_to_dpo
from resolvelm.data.generator import generate_corpus, gold_analysis
from resolvelm.data.split import grouped_temporal_split


def test_preference_corruption_categories() -> None:
    rec = grouped_temporal_split(generate_corpus(4, 1))[0]
    chosen = gold_analysis(rec)
    invalid = _corrupt(rec, "invalid_json", chosen)
    assert "{" in invalid
    assert "not json" in invalid
    wrong = json.loads(_corrupt(rec, "wrong_root_cause", chosen))
    assert wrong["root_cause_code"] != rec.root_cause_code
    missing = json.loads(_corrupt(rec, "missing_evidence", chosen))
    assert missing["evidence"] == []
    danger = json.loads(_corrupt(rec, "dangerous_remediation", chosen))
    assert danger["recommended_actions"][0]["action_id"] == "drop_production_db"
    unsup = json.loads(_corrupt(rec, "unsupported_evidence", chosen))
    assert unsup["evidence"][0]["source_id"] == "log-999"


def test_dpo_pairs_cover_train_only() -> None:
    records = grouped_temporal_split(generate_corpus(12, 2))
    train = [r for r in records if r.split == "train"]
    pairs = incidents_to_dpo(train, pairs_per_record=2)
    assert len(pairs) == len(train) * 2
