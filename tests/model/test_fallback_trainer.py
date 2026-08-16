from pathlib import Path

from resolvelm.data.convert import incidents_to_dpo, incidents_to_sft
from resolvelm.data.generator import generate_corpus
from resolvelm.data.split import grouped_temporal_split
from resolvelm.training.fallback import train_classifier_dpo, train_classifier_sft
from resolvelm.utils.io import write_jsonl


def test_fallback_sft_writes_loss(tmp_path: Path) -> None:
    records = grouped_temporal_split(generate_corpus(12, 1))
    train = [r for r in records if r.split == "train"]
    path = tmp_path / "sft.jsonl"
    write_jsonl(path, incidents_to_sft(train))
    summary = train_classifier_sft(path, tmp_path / "out", max_samples=12)
    assert summary["executed"] is True
    assert summary["final_train_loss"] >= 0
    assert (tmp_path / "out" / "classifier.joblib").exists()
    assert (tmp_path / "out" / "loss_curve.json").exists()


def test_fallback_dpo_writes_loss(tmp_path: Path) -> None:
    records = grouped_temporal_split(generate_corpus(12, 1))
    train = [r for r in records if r.split == "train"]
    path = tmp_path / "dpo.jsonl"
    write_jsonl(path, incidents_to_dpo(train, pairs_per_record=1))
    summary = train_classifier_dpo(path, tmp_path / "out", max_samples=8)
    assert summary["executed"] is True
    assert 0 <= summary["final_preference_accuracy"] <= 1
