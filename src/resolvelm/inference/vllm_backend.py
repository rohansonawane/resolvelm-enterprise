from __future__ import annotations


class VLLMBackend:
    """vLLM is not supported on this Apple Silicon development host."""

    name = "vllm"
    version = "vllm-unavailable"

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "VLLMBackend is not supported on this platform (no CUDA / Docker GPU). "
            "Use TransformersBackend or RuleBasedBackend locally. "
            "Production CUDA serving should use vLLM with Qwen3-4B-Instruct-2507-class models."
        )

    async def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        raise RuntimeError("VLLMBackend is not supported on this platform")

    def generate_sync(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        raise RuntimeError("VLLMBackend is not supported on this platform")
