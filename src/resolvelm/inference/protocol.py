from __future__ import annotations

from typing import Protocol


class ModelBackend(Protocol):
    name: str
    version: str

    async def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        """Return raw model text. Must not coerce invalid JSON into a valid analysis."""
        ...

    def generate_sync(self, prompt: str, *, max_new_tokens: int = 512) -> str: ...
