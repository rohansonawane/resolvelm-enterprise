"""Load a causal LM for SFT/DPO: full precision on CPU/MPS, 4-bit QLoRA on CUDA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from resolvelm.utils.device import detect_device


def quantization_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Return from_pretrained kwargs. Does not download weights."""
    import torch

    lora = config.get("lora") or {}
    train = config.get("train") or {}
    device_info = detect_device(train.get("device", "auto"))
    precision = str(train.get("precision", "fp32")).lower()
    dtype = (
        torch.bfloat16
        if precision == "bf16"
        else torch.float16
        if precision == "fp16"
        else torch.float32
    )
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool((config.get("model") or {}).get("trust_remote_code", False)),
        "torch_dtype": dtype,
    }
    if lora.get("qlora") and device_info["device"] == "cuda":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(lora.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(lora.get("double_quant", True)),
            bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.float16,
        )
        kwargs["device_map"] = "auto"
    return kwargs


def load_causal_lm(config: dict[str, Any], adapter_path: str | Path | None = None):
    model_id = config["model"]["id"]
    train = config.get("train") or {}
    lora = config.get("lora") or {}
    device_info = detect_device(train.get("device", "auto"))
    device = device_info["device"]
    if str(train.get("device", "")).lower() == "cuda" and device != "cuda":
        raise RuntimeError(
            "CUDA was requested but is not available. Install a CUDA torch wheel. See GPU_SETUP.md."
        )
    if lora.get("qlora") and device != "cuda":
        raise RuntimeError("QLoRA (4-bit) requires CUDA. See GPU_SETUP.md.")

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=bool(kwargs_trust(config))
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    extra = quantization_kwargs(config)
    model = AutoModelForCausalLM.from_pretrained(model_id, **extra)
    peft_config = LoraConfig(
        r=int(lora.get("r", 8)),
        lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora.get("target_modules", ["q_proj", "v_proj"])),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    adapter = Path(adapter_path) if adapter_path else None
    if adapter and adapter.exists():
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)

    if device in {"mps", "cpu"} and "device_map" not in extra:
        model.to(torch.device(device))
    if bool(train.get("gradient_checkpointing")):
        model.enable_input_require_grads()
    return tokenizer, model, device_info, peft_config


def kwargs_trust(config: dict[str, Any]) -> bool:
    return bool((config.get("model") or {}).get("trust_remote_code", False))


def trainer_precision_flags(config: dict[str, Any]) -> dict[str, bool]:
    precision = str((config.get("train") or {}).get("precision", "fp32")).lower()
    device = detect_device((config.get("train") or {}).get("device", "auto"))["device"]
    if device != "cuda":
        return {"fp16": False, "bf16": False}
    return {"fp16": precision == "fp16", "bf16": precision == "bf16"}
