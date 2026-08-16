from __future__ import annotations

import shutil
from typing import Any


def free_disk_gb(path: str = "/") -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def detect_device(preferred: str = "auto") -> dict[str, Any]:
    info: dict[str, Any] = {
        "preferred": preferred,
        "torch_available": False,
        "cuda": False,
        "mps": False,
        "device": "cpu",
        "torch_version": None,
    }
    try:
        import torch

        info["torch_available"] = True
        info["torch_version"] = str(torch.__version__)
        info["cuda"] = bool(torch.cuda.is_available())
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        info["mps"] = mps
        if preferred == "cuda" and info["cuda"]:
            info["device"] = "cuda"
        elif preferred == "mps" and mps:
            info["device"] = "mps"
        elif preferred in {"cpu", "mps", "cuda"}:
            info["device"] = preferred if (preferred != "cuda" or info["cuda"]) else "cpu"
            if preferred == "mps" and not mps:
                info["device"] = "cpu"
        elif info["cuda"]:
            info["device"] = "cuda"
        elif mps:
            info["device"] = "mps"
        else:
            info["device"] = "cpu"
    except Exception:
        info["device"] = "cpu"
    return info
