from __future__ import annotations

from pathlib import Path
from typing import Any

from resolvelm.inference.classifier_backend import ClassifierBackend
from resolvelm.inference.protocol import ModelBackend
from resolvelm.inference.rule_backend import RuleBasedBackend
from resolvelm.inference.vllm_backend import VLLMBackend


def load_backend(spec: dict[str, Any]) -> ModelBackend:
    kind = spec.get("kind", "rule")
    if kind == "rule":
        return RuleBasedBackend()
    if kind == "vllm":
        return VLLMBackend()
    if kind == "classifier":
        path = spec["classifier_path"]
        if not Path(path).exists():
            raise FileNotFoundError(f"classifier artifact missing: {path}")
        return ClassifierBackend(path)
    if kind == "transformers":
        from resolvelm.inference.transformers_backend import TransformersBackend

        return TransformersBackend(
            model_id=spec["model_id"],
            adapter_path=spec.get("adapter_path"),
            device=spec.get("device", "auto"),
            load_in_4bit=bool(spec.get("load_in_4bit", False)),
        )
    if kind == "auto":
        adapter = spec.get("adapter_path")
        classifier = spec.get("classifier_path")
        if classifier and Path(classifier).exists():
            return ClassifierBackend(classifier)
        if adapter and (Path(adapter) / "adapter_config.json").exists():
            from resolvelm.inference.transformers_backend import TransformersBackend

            return TransformersBackend(
                model_id=spec.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct"),
                adapter_path=adapter,
                device=spec.get("device", "auto"),
                load_in_4bit=bool(spec.get("load_in_4bit", False)),
            )
        return RuleBasedBackend()
    raise ValueError(f"unknown backend kind: {kind}")
