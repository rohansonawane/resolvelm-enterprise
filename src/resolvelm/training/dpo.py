"""DPO entrypoint: TRL DPOTrainer when possible, preference SGD fallback otherwise."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from resolvelm.training.fallback import train_classifier_dpo
from resolvelm.training.mlflow_utils import try_mlflow_log
from resolvelm.utils.device import free_disk_gb
from resolvelm.utils.io import write_json
from resolvelm.utils.seed import set_global_seed


def _run_trl_dpo(config: dict[str, Any], train_path: Path, output_dir: Path) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    from resolvelm.training.hf_load import load_causal_lm, trainer_precision_flags

    model_id = config["model"]["id"]
    train_cfg = config["train"]
    smoke = config.get("smoke") or {}
    adapter = config.get("sft_adapter")
    tokenizer, model, device_info, peft_config = load_causal_lm(config, adapter_path=adapter)
    ds = load_dataset("json", data_files={"train": str(train_path)})
    if smoke.get("max_train_samples"):
        n = min(int(smoke["max_train_samples"]), len(ds["train"]))
        ds["train"] = ds["train"].select(range(n))
    max_steps = int(smoke.get("max_steps", 0) or 0)
    already_peft = bool(adapter and Path(adapter).exists())
    args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("grad_accum", 1)),
        learning_rate=float(train_cfg.get("learning_rate", 5e-5)),
        num_train_epochs=float(train_cfg.get("epochs", 1)),
        max_steps=max_steps if max_steps > 0 else -1,
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        beta=float(train_cfg.get("beta", 0.1)),
        max_length=int(train_cfg.get("max_seq_len", 512)),
        max_prompt_length=int(train_cfg.get("max_seq_len", 512)) // 2,
        report_to=[],
        seed=int(config.get("seed", 42)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        remove_unused_columns=False,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
        **trainer_precision_flags(config),
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        processing_class=tokenizer,
        peft_config=None if already_peft else peft_config,
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = result.metrics
    return {
        "backend": "trl_dpo",
        "executed": True,
        "transformers_used": True,
        "model_id": model_id,
        "device": device_info["device"],
        "qlora": bool((config.get("lora") or {}).get("qlora")),
        "sft_adapter": str(adapter) if adapter else None,
        "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
        "train_loss": metrics.get("train_loss"),
        "torch": torch.__version__,
    }


def run_dpo(config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    import os

    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = root or Path.cwd()
    set_global_seed(int(config.get("seed", 42)))
    train_path = root / config["data"]["train_path"]
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = root / config.get("artifacts_dir", "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    smoke = config.get("smoke") or {}
    disk = free_disk_gb()
    min_disk = float((config.get("model") or {}).get("min_free_disk_gb", 6.0))
    allow_hf = bool((config.get("model") or {}).get("allow_download", True))
    strict_cuda = str((config.get("train") or {}).get("device", "")).lower() == "cuda" or str(
        config.get("profile", "")
    ).startswith("cuda")
    try:
        if not (allow_hf and disk >= min_disk):
            raise RuntimeError(f"skip HF DPO disk={disk:.2f}")
        # DPO + LoRA on 0.5B can OOM on 16GB; still attempt, then fall back unless CUDA.
        summary = _run_trl_dpo(config, train_path, output_dir)
    except Exception as exc:  # noqa: BLE001
        if strict_cuda:
            raise
        fallback = train_classifier_dpo(
            train_path,
            output_dir,
            seed=int(config.get("seed", 42)),
            max_samples=smoke.get("max_train_samples"),
            name="dpo",
        )
        summary = {**fallback, "hf_error": str(exc), "free_disk_gb": disk}
    write_json(artifacts_dir / "dpo_train_metrics.json", summary)
    try_mlflow_log(
        {**config, "run_name": "dpo", "register_alias": config.get("register_alias", "champion")},
        summary.get("metrics") or {"train_loss": summary.get("final_train_loss", 0)},
        artifacts_dir,
    )
    return summary
