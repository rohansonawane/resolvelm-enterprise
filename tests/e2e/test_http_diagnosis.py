from fastapi.testclient import TestClient

from resolvelm.api.app import create_app
from resolvelm.evaluation.demos import demo_incidents
from resolvelm.inference.rule_backend import RuleBasedBackend


def test_e2e_enter_incident_get_diagnosis() -> None:
    app = create_app()
    app.state.backend = RuleBasedBackend()
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    ready = client.get("/ready")
    assert ready.status_code == 200
    demo = next(d for d in demo_incidents() if d["demo_id"] == "disk_full")
    response = client.post("/api/v1/incidents/analyze", json=demo["request"])
    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert analysis["root_cause_code"] == "DISK_FULL"
    assert analysis["recommended_actions"]
    models = client.get("/api/v1/models")
    assert models.status_code == 200
    assert models.json()["production_target"]["executed_locally"] is False
