from fastapi.testclient import TestClient

from resolvelm.api.app import create_app
from resolvelm.evaluation.demos import demo_incidents
from resolvelm.inference.rule_backend import RuleBasedBackend


def test_analyze_roundtrip() -> None:
    app = create_app()
    app.state.backend = RuleBasedBackend()
    client = TestClient(app)
    demo = demo_incidents()[3]["request"]  # connection exhaustion
    response = client.post("/api/v1/incidents/analyze", json=demo)
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    assert body["analysis"]["root_cause_code"] == "DB_CONNECTION_POOL_EXHAUSTION"
    assert body["analysis"]["severity"] == "SEV-2"
    assert body["error"] is None


def test_batch_analyze() -> None:
    app = create_app()
    client = TestClient(app)
    demos = [d["request"] for d in demo_incidents()[:2]]
    response = client.post("/api/v1/incidents/batch-analyze", json={"incidents": demos})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2
