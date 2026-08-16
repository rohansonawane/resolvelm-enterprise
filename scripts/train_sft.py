#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from resolvelm.training.sft import run_sft
from resolvelm.utils.config import load_yaml, project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local/sft.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = project_root()
    cfg = load_yaml(root / args.config)
    if args.smoke:
        cfg.setdefault("smoke", {})
        cfg["smoke"].setdefault("max_train_samples", 16)
        cfg["smoke"].setdefault("max_steps", 3)
    summary = run_sft(cfg, root=root)
    print(json.dumps({k: summary[k] for k in summary if k != "loss_curve"}, indent=2, default=str))


if __name__ == "__main__":
    main()
