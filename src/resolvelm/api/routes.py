from __future__ import annotations

import asyncio
import json
import time

# json is used by /api/v1/benchmarks and /api/v1/experiments
from fastapi import APIRouter, HTTPException, Request

from resolvelm import __version__
from resolvelm.api.settings import get_settings
from resolvelm.evaluation.demos import demo_incidents
from resolvelm.inference.prompting import request_to_prompt
from resolvelm.inference.repair import parse_analysis_with_retry
from resolvelm.monitoring.metrics import PARSE_FAILURES, metrics_output
from resolvelm.schemas.api import (
    AnalysisErrorPayload,
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    HealthResponse,
)
from resolvelm.utils.config import project_root
from resolvelm.utils.device import detect_device

router = APIRouter()


def _backend(request: Request):
    backend = getattr(request.app.state, "backend", None)
    if backend is None:
        raise HTTPException(status_code=503, detail="model backend is not loaded")
    return backend


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = get_settings()
    device = detect_device(settings.device)
    backend = getattr(request.app.state, "backend", None)
    return HealthResponse(
        status="ok" if backend is not None else "degraded",
        version=__version__,
        backend=getattr(backend, "name", "none"),
        device=str(device["device"]),
    )


@router.get("/ready")
async def ready(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "backend", None) is None:
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ready"}


@router.get("/metrics")
async def metrics() -> bytes:
    body, _ctype = metrics_output()
    return body


@router.get("/api/v1/models")
async def models(request: Request) -> dict:
    settings = get_settings()
    backend = getattr(request.app.state, "backend", None)
    return {
        "local": {
            "name": getattr(backend, "name", "none"),
            "version": getattr(backend, "version", settings.model_version),
            "backend": getattr(backend, "name", "none"),
            "executed_locally": True,
        },
        "production_target": {
            "name": settings.production_model_id,
            "backend": "vllm",
            "executed_locally": False,
            "note": "Not benchmarked on this hardware.",
        },
    }


@router.get("/api/v1/benchmarks")
async def benchmarks() -> dict:
    root = project_root()
    comparison = root / "artifacts" / "model_comparison.csv"
    metrics_path = root / "artifacts" / "base_model_metrics.json"
    payload: dict = {"available": False, "rows": [], "note": "Run make smoke or make evaluate."}
    if metrics_path.exists():
        payload["base"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["available"] = True
    if comparison.exists():
        import csv

        with comparison.open("r", encoding="utf-8") as handle:
            payload["rows"] = list(csv.DictReader(handle))
        payload["available"] = True
    return payload


@router.get("/api/v1/demos")
async def demos() -> dict:
    return {"demos": demo_incidents()}


@router.get("/api/v1/experiments")
async def experiments() -> dict:
    root = project_root()
    artifacts = {}
    for name in ("sft_train_metrics.json", "dpo_train_metrics.json", "environment.json"):
        path = root / "artifacts" / name
        if path.exists():
            artifacts[name] = json.loads(path.read_text(encoding="utf-8"))
    return {"artifacts": artifacts, "mlflow_note": "Start MLflow on :5101 to browse runs."}


async def _analyze_one(request: Request, body: AnalyzeRequest, request_id: str) -> AnalyzeResponse:
    settings = get_settings()
    backend = _backend(request)
    prompt = request_to_prompt(body)
    t0 = time.perf_counter()
    try:
        raw = await asyncio.wait_for(
            backend.generate(prompt),
            timeout=settings.generate_timeout_s,
        )
    except TimeoutError:
        return AnalyzeResponse(
            request_id=request_id,
            model_version=getattr(backend, "version", settings.model_version),
            backend=backend.name,
            latency_ms=(time.perf_counter() - t0) * 1000,
            error=AnalysisErrorPayload(
                error="timeout",
                message=f"generation exceeded {settings.generate_timeout_s}s",
            ),
            needs_human_review=True,
        )
    except Exception as exc:  # noqa: BLE001
        return AnalyzeResponse(
            request_id=request_id,
            model_version=getattr(backend, "version", settings.model_version),
            backend=backend.name,
            latency_ms=(time.perf_counter() - t0) * 1000,
            error=AnalysisErrorPayload(
                error="backend_unavailable",
                message=str(exc),
            ),
            needs_human_review=True,
        )
    parsed = parse_analysis_with_retry(raw)
    latency = (time.perf_counter() - t0) * 1000
    if not parsed.ok:
        PARSE_FAILURES.inc()
        return AnalyzeResponse(
            request_id=request_id,
            model_version=getattr(backend, "version", settings.model_version),
            backend=backend.name,
            latency_ms=latency,
            error=AnalysisErrorPayload(
                error="invalid_model_output",
                message=parsed.error or "model output failed schema validation",
                parse_attempts=parsed.attempts,
                raw_output_preview=parsed.raw[:400],
            ),
            needs_human_review=True,
        )
    analysis = parsed.analysis
    assert analysis is not None
    known = {e.event_id for e in body.log_events + body.metric_events + body.trace_events}
    if known and any(ev.source_id not in known for ev in analysis.evidence):
        analysis = analysis.model_copy(update={"needs_human_review": True})
    return AnalyzeResponse(
        request_id=request_id,
        model_version=getattr(backend, "version", settings.model_version),
        backend=backend.name,
        latency_ms=latency,
        analysis=analysis,
        needs_human_review=analysis.needs_human_review,
    )


@router.post("/api/v1/incidents/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, body: AnalyzeRequest) -> AnalyzeResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    return await _analyze_one(request, body, request_id)


@router.post("/api/v1/incidents/batch-analyze", response_model=BatchAnalyzeResponse)
async def batch_analyze(request: Request, body: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    results = []
    for i, item in enumerate(body.incidents):
        results.append(await _analyze_one(request, item, f"{request_id}:{i}"))
    return BatchAnalyzeResponse(request_id=request_id, results=results)
