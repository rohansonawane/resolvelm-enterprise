from resolvelm.evaluation.metrics import classification_report_safe, latency_stats, set_prf


def test_metrics_helpers() -> None:
    report = classification_report_safe(["A", "B", "A"], ["A", "A", "A"])
    assert 0 <= report["accuracy"] <= 1
    assert report["labels"]
    lat = latency_stats([1.0, 2.0, 3.0, 4.0])
    assert lat["n"] == 4
    prf = set_prf({"a", "b"}, {"a"})
    assert 0 < prf["f1"] < 1
