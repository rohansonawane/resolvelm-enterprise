#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from resolvelm.training.dpo import run_dpo
from resolvelm.utils.config import load_yaml, project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local/dpo.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = project_root()
    cfg = load_yaml(root / args.config)
    if args.smoke:
        cfg.setdefault("smoke", {})
        cfg["smoke"].setdefault("max_train_samples", 8)
        cfg["smoke"].setdefault("max_steps", 2)
    summary = run_dpo(cfg, root=root)
    print(json.dumps({k: summary[k] for k in summary if k != "loss_curve"}, indent=2, default=str))


if __name__ == "__main__":
    main()
