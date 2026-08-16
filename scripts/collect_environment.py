#!/usr/bin/env python3
from __future__ import annotations

import platform
import sys
from pathlib import Path

from resolvelm import __version__
from resolvelm.utils.config import project_root
from resolvelm.utils.device import detect_device, free_disk_gb
from resolvelm.utils.io import write_json


def main() -> None:
    root = project_root()
    info = {
        "project": "resolvelm",
        "version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "free_disk_gb": free_disk_gb(),
        "device": detect_device("auto"),
        "cuda_executed": False,
        "qwen3_4b_executed": False,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_mps"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception as exc:  # noqa: BLE001
        info["torch"] = None
        info["torch_error"] = str(exc)
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception as exc:  # noqa: BLE001
        info["transformers"] = None
        info["transformers_error"] = str(exc)
    write_json(root / "artifacts" / "environment.json", info)
    print(Path(root / "artifacts" / "environment.json"))


if __name__ == "__main__":
    main()
