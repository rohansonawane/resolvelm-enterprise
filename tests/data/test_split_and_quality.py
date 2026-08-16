from resolvelm.data.generator import generate_corpus
from resolvelm.data.leakage import family_leakage
from resolvelm.data.split import grouped_temporal_split
from resolvelm.data.validate import validate_incidents


def test_grouped_split_keeps_families_together() -> None:
    records = grouped_temporal_split(generate_corpus(18, 2))
    families = {}
    for rec in records:
        families.setdefault(rec.family_id, set()).add(rec.split)
    assert all(len(v) == 1 for v in families.values())
    train = [r for r in records if r.split == "train"]
    test = [r for r in records if r.split == "test"]
    assert family_leakage(train, test) == []


def test_quality_report_counts() -> None:
    records = grouped_temporal_split(generate_corpus(12, 2))
    report = validate_incidents(records)
    assert report["n_records"] == 24
    assert "label_origin_counts" in report
