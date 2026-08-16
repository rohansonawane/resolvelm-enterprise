"""SFT entrypoint: TRL + PEFT LoRA when possible, sklearn fallback otherwise."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from resolvelm.training.fallback import train_classifier_sft
from resolvelm.training.mlflow_utils import try_mlflow_log
from resolvelm.utils.device import detect_device, free_disk_gb
from resolvelm.utils.io import write_json
from resolvelm.utils.seed import set_global_seed


def _maybe_optuna(config: dict[str, Any], train_path: Path) -> dict[str, Any]:
    opt = config.get("optuna") or {}
    if not opt.get("enabled"):
        return {"skipped": True, "reason": "optuna.enabled is false"}
    try:
        import optuna
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        from resolvelm.utils.io import read_jsonl

        rows = read_jsonl(train_path)[: int(opt.get("subset_size", 16))]
        texts = [r.get("text") or str(r.get("messages")) for r in rows]
        labels = [r["root_cause_code"] for r in rows]
        if len(set(labels)) < 2 or len(texts) < 4:
            return {"skipped": True, "reason": "subset too small for Optuna"}

        def objective(trial: optuna.Trial) -> float:
            c = trial.suggest_float("C", 0.1, 5.0, log=True)
            vec = TfidfVectorizer(max_features=trial.suggest_int("max_features", 256, 2048))
            x = vec.fit_transform(texts)
            clf = LogisticRegression(C=c, max_iter=120)
            scores = cross_val_score(clf, x, labels, cv=2)
            return float(scores.mean())

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective, n_trials=int(opt.get("n_trials", 2)), timeout=opt.get("timeout_s")
        )
        return {
            "skipped": False,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        }
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": str(exc)}


def _run_trl_sft(
    config: dict[str, Any], train_path: Path, val_path: Path, output_dir: Path
) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    model_id = config["model"]["id"]
    train_cfg = config["train"]
    lora_cfg = config["lora"]
    smoke = config.get("smoke") or {}
    max_steps = int(smoke.get("max_steps", 0) or 0)
    device_info = detect_device(train_cfg.get("device", "auto"))
    device = device_info["device"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    if device == "mps":
        model.to(torch.device("mps"))

    data_files = {"train": str(train_path)}
    if val_path.exists():
        data_files["validation"] = str(val_path)
    ds = load_dataset("json", data_files=data_files)
    if smoke.get("max_train_samples"):
        n = min(int(smoke["max_train_samples"]), len(ds["train"]))
        ds["train"] = ds["train"].select(range(n))
    if "validation" in ds and smoke.get("max_val_samples"):
        n = min(int(smoke["max_val_samples"]), len(ds["validation"]))
        ds["validation"] = ds["validation"].select(range(n))

    class NanAbort(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
            if logs and any(
                isinstance(v, float) and (v != v or v == float("inf")) for v in logs.values()
            ):
                raise RuntimeError(f"NaN/Inf loss abort: {logs}")

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("grad_accum", 1)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        num_train_epochs=float(train_cfg.get("epochs", 1)),
        max_steps=max_steps if max_steps > 0 else -1,
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        save_total_limit=int(train_cfg.get("save_total_limit", 1)),
        max_seq_length=int(train_cfg.get("max_seq_len", 512)),
        packing=False,
        report_to=[],
        seed=int(config.get("seed", 42)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        lr_scheduler_type="cosine",
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        dataset_text_field="text",
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
        fp16=False,
        bf16=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        processing_class=tokenizer,
        callbacks=[NanAbort()],
    )
    try:
        result = trainer.train(resume_from_checkpoint=bool(train_cfg.get("resume")))
    except torch.cuda.OutOfMemoryError:
        sft_args.per_device_train_batch_size = 1
        sft_args.gradient_accumulation_steps = 1
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=ds["train"],
            processing_class=tokenizer,
        )
        result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    history = result.metrics
    return {
        "backend": "trl_sft",
        "executed": True,
        "transformers_used": True,
        "device": device,
        "model_id": model_id,
        "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in history.items()},
        "train_loss": history.get("train_loss"),
    }


def run_sft(config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    import os

    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = root or Path.cwd()
    set_global_seed(int(config.get("seed", 42)))
    train_path = root / config["data"]["train_path"]
    val_path = root / config["data"]["val_path"]
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = root / config.get("artifacts_dir", "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    smoke = config.get("smoke") or {}
    max_samples = smoke.get("max_train_samples")

    optuna_result = _maybe_optuna(config, train_path)
    device_info = detect_device((config.get("train") or {}).get("device", "auto"))
    disk = free_disk_gb()
    allow_hf = bool((config.get("model") or {}).get("allow_download", True))
    min_disk = float((config.get("model") or {}).get("min_free_disk_gb", 6.0))
    use_hf = allow_hf and disk >= min_disk

    summary: dict[str, Any]
    try:
        if not use_hf:
            raise RuntimeError(f"skip HF: disk={disk:.2f}GiB < {min_disk} or download disabled")
        summary = _run_trl_sft(config, train_path, val_path, output_dir)
    except Exception as exc:  # noqa: BLE001
        fallback = train_classifier_sft(
            train_path,
            output_dir,
            seed=int(config.get("seed", 42)),
            max_samples=max_samples,
            name="sft",
        )
        summary = {
            **fallback,
            "hf_error": str(exc),
            "device_info": device_info,
            "free_disk_gb": disk,
        }
    summary["optuna"] = optuna_result
    summary["lora"] = config.get("lora")
    summary["seed"] = config.get("seed")
    write_json(artifacts_dir / "sft_train_metrics.json", summary)
    try_mlflow_log(
        config,
        summary.get("metrics") or {"train_loss": summary.get("final_train_loss", 0)},
        artifacts_dir,
    )
    return summary
