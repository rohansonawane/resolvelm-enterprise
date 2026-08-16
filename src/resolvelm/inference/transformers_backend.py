from __future__ import annotations

from pathlib import Path
from typing import Any

from resolvelm.utils.device import detect_device


class TransformersBackend:
    name = "transformers"
    version = "transformers-unloaded"

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        device: str = "auto",
        max_new_tokens: int = 512,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.version = adapter_path or model_id
        info = detect_device(device)
        self.device = info["device"]
        self._model = None
        self._tokenizer = None
        self._load()

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        extra: dict[str, Any] = {}
        if self.load_in_4bit:
            if self.device != "cuda":
                raise RuntimeError("load_in_4bit requires CUDA. See GPU_SETUP.md.")
            import torch
            from transformers import BitsAndBytesConfig

            extra["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            extra["device_map"] = "auto"
        tok = AutoTokenizer.from_pretrained(self.model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **extra)
        if self.adapter_path and Path(self.adapter_path).exists():
            from peft import PeftModel

            adapter_file = Path(self.adapter_path)
            if (adapter_file / "adapter_config.json").exists() or (
                adapter_file / "adapter_model.safetensors"
            ).exists():
                model = PeftModel.from_pretrained(model, self.adapter_path)
        if "device_map" not in extra:
            model.to(self.device)
        model.eval()
        self._tokenizer = tok
        self._model = model
        self.version = f"{self.model_id}:{self.adapter_path or 'base'}"

    def generate_sync(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("TransformersBackend is not loaded")
        tokens = min(max_new_tokens, self.max_new_tokens)
        tokenizer = self._tokenizer
        encoded: Any = tokenizer(prompt, return_tensors="pt")
        target = next(self._model.parameters()).device
        encoded = {k: v.to(target) for k, v in encoded.items()}
        with torch.no_grad():
            out = self._model.generate(
                **encoded,
                max_new_tokens=tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        return text

    async def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        return self.generate_sync(prompt, max_new_tokens=max_new_tokens)
