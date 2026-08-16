"""Training callbacks kept as importable helpers for TRL/Transformers."""

from __future__ import annotations

from typing import Any


class NanAbortCallback:
    def on_log(self, logs: dict[str, Any] | None) -> None:
        if not logs:
            return
        for key, value in logs.items():
            if isinstance(value, float) and (value != value or value == float("inf")):
                raise RuntimeError(f"NaN/Inf detected in {key}={value}")
