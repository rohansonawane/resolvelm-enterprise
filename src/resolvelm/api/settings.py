from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RESOLVELM_", extra="ignore")

    env: str = "local"
    api_host: str = "0.0.0.0"
    api_port: int = 8101
    log_level: str = "INFO"
    max_payload_bytes: int = 262144
    generate_timeout_s: float = 30.0
    backend: str = "rule"
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path: str = ""
    classifier_path: str = ""
    model_version: str = "rule-baseline-v1"
    device: str = "auto"
    production_model_id: str = "Qwen/Qwen3-4B-Instruct"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def backend_spec_from_env() -> dict:
    settings = get_settings()
    kind = settings.backend
    spec = {
        "kind": kind,
        "model_id": settings.model_id,
        "adapter_path": settings.adapter_path or None,
        "classifier_path": settings.classifier_path or os.environ.get("RESOLVELM_CLASSIFIER_PATH"),
        "device": settings.device,
    }
    return spec
