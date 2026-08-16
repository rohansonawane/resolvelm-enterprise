"""Optional MLX interface. Not implemented as a serving backend on this host."""

from __future__ import annotations


class MLXBackend:
    name = "mlx"
    version = "mlx-interface-only"

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "MLXBackend is an optional interface only. ResolveLM serving uses "
            "TransformersBackend, RuleBasedBackend, or VLLMBackend (CUDA)."
        )

    async def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        raise RuntimeError("MLXBackend is not implemented")

    def generate_sync(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        raise RuntimeError("MLXBackend is not implemented")
