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
    ) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.version = adapter_path or model_id
        info = detect_device(device)
        self.device = info["device"]
        self._model = None
        self._tokenizer = None
        self._load()

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        if self.adapter_path and Path(self.adapter_path).exists():
            from peft import PeftModel

            adapter_file = Path(self.adapter_path)
            if (adapter_file / "adapter_config.json").exists() or (
                adapter_file / "adapter_model.safetensors"
            ).exists():
                model = PeftModel.from_pretrained(model, self.adapter_path)
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
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
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
