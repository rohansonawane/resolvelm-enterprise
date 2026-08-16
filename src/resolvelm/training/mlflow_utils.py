from __future__ import annotations

from pathlib import Path
from typing import Any

from resolvelm.utils.io import write_json


def try_mlflow_log(
    config: dict[str, Any], metrics: dict[str, Any], artifact_dir: Path
) -> dict[str, Any]:
    info: dict[str, Any] = {"enabled": False, "error": None, "run_id": None}
    try:
        import mlflow

        tracking = config.get("mlflow_tracking_uri")
        if tracking:
            mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment(config.get("mlflow_experiment", "resolvelm"))
        with mlflow.start_run(run_name=config.get("run_name", "resolvelm")) as run:
            params = {
                "seed": config.get("seed"),
                "model_id": (config.get("model") or {}).get("id"),
                "lr": (config.get("train") or {}).get("learning_rate"),
                "lora_r": (config.get("lora") or {}).get("r"),
                "lora_alpha": (config.get("lora") or {}).get("alpha"),
                "lora_dropout": (config.get("lora") or {}).get("dropout"),
                "batch_size": (config.get("train") or {}).get("batch_size"),
                "grad_accum": (config.get("train") or {}).get("grad_accum"),
                "max_seq_len": (config.get("train") or {}).get("max_seq_len"),
                "epochs": (config.get("train") or {}).get("epochs"),
                "optimizer": (config.get("train") or {}).get("optimizer"),
                "precision": (config.get("train") or {}).get("precision"),
                "device": (config.get("train") or {}).get("device"),
                "dataset_version": config.get("dataset_version", "fixture-v1"),
            }
            mlflow.log_params({k: str(v) for k, v in params.items() if v is not None})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
            if artifact_dir.exists():
                mlflow.log_artifacts(str(artifact_dir), artifact_path="training")
            alias = config.get("register_alias")
            if alias:
                try:
                    mlflow.set_tag("registry_alias", alias)
                except Exception as exc:  # noqa: BLE001
                    info["register_error"] = str(exc)
            info["enabled"] = True
            info["run_id"] = run.info.run_id
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    write_json(artifact_dir / "mlflow_log.json", info)
    return info
