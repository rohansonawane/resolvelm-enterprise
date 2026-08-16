#!/usr/bin/env python3
"""Latency smoke for the loaded backend. Does not invent GPU throughput."""

from __future__ import annotations

import argparse
import json
import time

from resolvelm.evaluation.demos import demo_incidents
from resolvelm.inference.prompting import request_to_prompt
from resolvelm.inference.registry import load_backend
from resolvelm.schemas.api import AnalyzeRequest
from resolvelm.utils.config import load_yaml, project_root
from resolvelm.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local/evaluation.yaml")
    args = parser.parse_args()
    root = project_root()
    cfg = load_yaml(root / args.config)
    backend = load_backend((cfg.get("backends") or {}).get("base") or {"kind": "rule"})
    latencies = []
    for demo in demo_incidents():
        req = AnalyzeRequest.model_validate(demo["request"])
        prompt = request_to_prompt(req)
        t0 = time.perf_counter()
        backend.generate_sync(prompt)
        latencies.append((time.perf_counter() - t0) * 1000)
    payload = {
        "backend": backend.name,
        "n": len(latencies),
        "latencies_ms": latencies,
        "p50_ms": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "tokens_per_sec": None,
        "peak_memory_mb": None,
        "note": "Not benchmarked on this hardware."
        if backend.name != "rule"
        else "Rule backend wall-clock latency only.",
    }
    write_json(root / "artifacts" / "benchmark.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
