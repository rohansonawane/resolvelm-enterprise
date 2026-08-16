import pytest
from fastapi.testclient import TestClient

from resolvelm.api.app import create_app
from resolvelm.evaluation.demos import demo_incidents
from resolvelm.inference.repair import parse_analysis_with_retry
from resolvelm.inference.vllm_backend import VLLMBackend


class BrokenBackend:
    name = "broken"
    version = "broken"

    async def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        return "this is not json at all"

    def generate_sync(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        return "this is not json at all"


class MissingBackend:
    name = "missing"
    version = "missing"

    async def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        raise RuntimeError("weights not found")

    def generate_sync(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        raise RuntimeError("weights not found")


def test_malformed_json_returns_error_payload() -> None:
    app = create_app()
    app.state.backend = BrokenBackend()
    client = TestClient(app)
    demo = demo_incidents()[0]["request"]
    response = client.post("/api/v1/incidents/analyze", json=demo)
    assert response.status_code == 200
    body = response.json()
    assert body["analysis"] is None
    assert body["error"]["error"] == "invalid_model_output"
    assert body["error"]["parse_attempts"] >= 1
    assert body["needs_human_review"] is True


def test_missing_model_error() -> None:
    app = create_app()
    app.state.backend = MissingBackend()
    client = TestClient(app)
    demo = demo_incidents()[0]["request"]
    response = client.post("/api/v1/incidents/analyze", json=demo)
    assert response.json()["error"]["error"] == "backend_unavailable"


def test_oversized_payload() -> None:
    app = create_app()
    client = TestClient(app)
    demo = demo_incidents()[0]["request"]
    demo = {**demo, "symptoms": ["x" * 200000]}
    response = client.post(
        "/api/v1/incidents/analyze",
        json=demo,
        headers={"content-length": str(500_000)},
    )
    assert response.status_code in {413, 422}


def test_vllm_raises_clear_error() -> None:
    with pytest.raises(RuntimeError, match="not supported"):
        VLLMBackend()


def test_parse_does_not_invent_valid_result() -> None:
    parsed = parse_analysis_with_retry('{"severity":"SEV-2"}')
    assert parsed.analysis is None


def test_generation_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from resolvelm.api.settings import get_settings

    class SlowBackend:
        name = "slow"
        version = "slow"

        async def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
            await asyncio.sleep(1)
            return "{}"

        def generate_sync(self, prompt: str, *, max_new_tokens: int = 512) -> str:
            return "{}"

    monkeypatch.setenv("RESOLVELM_GENERATE_TIMEOUT_S", "0.01")
    get_settings.cache_clear()
    app = create_app()
    app.state.backend = SlowBackend()
    client = TestClient(app)
    demo = demo_incidents()[0]["request"]
    response = client.post("/api/v1/incidents/analyze", json=demo)
    get_settings.cache_clear()
    assert response.json()["error"]["error"] == "timeout"
