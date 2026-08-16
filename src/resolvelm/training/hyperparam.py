from __future__ import annotations

from typing import Any


def search_space(trial: Any) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
        "lora_r": trial.suggest_categorical("lora_r", [4, 8, 16]),
        "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.1),
    }
