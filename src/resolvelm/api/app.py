from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from resolvelm.api.middleware import RequestContextMiddleware
from resolvelm.api.routes import router
from resolvelm.api.settings import backend_spec_from_env, get_settings
from resolvelm.inference.registry import load_backend
from resolvelm.monitoring.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="ResolveLM",
        description="Enterprise LLM post-training platform for incident resolution",
        version="0.1.0",
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3101", "http://127.0.0.1:3101"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    try:
        app.state.backend = load_backend(backend_spec_from_env())
    except Exception:
        from resolvelm.inference.rule_backend import RuleBasedBackend

        app.state.backend = RuleBasedBackend()
    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "resolvelm.api.app:app", host=settings.api_host, port=settings.api_port, reload=False
    )
