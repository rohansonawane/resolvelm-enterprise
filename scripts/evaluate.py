#!/usr/bin/env python3
"""Evaluate BASE vs SFT vs DPO on the locked test split and write artifacts."""

from __future__ import annotations

import argparse
import csv
import json

from resolvelm.evaluation.compare import comparison_rows
from resolvelm.evaluation.error_analysis import collect_failures
from resolvelm.evaluation.evaluator import evaluate_backend
from resolvelm.inference.registry import load_backend
from resolvelm.schemas.incident import IncidentRecord
from resolvelm.utils.config import load_yaml, project_root
from resolvelm.utils.io import read_jsonl, write_json, write_jsonl


def _md_table(rows: list[dict]) -> str:
    if not rows:
        return "_No comparison rows._\n"
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[k])[:24] for k in keys) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local/evaluation.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--docs-only", action="store_true")
    args = parser.parse_args()
    root = project_root()
    cfg = load_yaml(root / args.config)
    artifacts = root / cfg.get("artifacts_dir", "artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    docs = root / cfg.get("docs_dir", "docs")
    docs.mkdir(parents=True, exist_ok=True)
    pred_dir = root / cfg.get("predictions_dir", "artifacts/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    if args.docs_only and (artifacts / "model_comparison.csv").exists():
        print("docs-only: comparison already present")
        return

    rows = read_jsonl(root / cfg["test_path"])
    max_samples = cfg.get("max_samples")
    if args.smoke:
        max_samples = max_samples or 24
    if max_samples:
        rows = rows[: int(max_samples)]
    records = [IncidentRecord.model_validate(r) for r in rows]

    suite = {}
    for key, spec in (cfg.get("backends") or {}).items():
        backend = load_backend(spec)
        result = evaluate_backend(backend, records, timeout_s=float(cfg.get("timeout_s", 30)))
        suite[key] = result
        write_json(artifacts / f"{key}_model_metrics.json", result["metrics"])
        write_jsonl(pred_dir / f"{key}_predictions.jsonl", result["predictions"])

    if "base" in suite:
        write_json(artifacts / "base_model_metrics.json", suite["base"]["metrics"])

    cmp_rows = comparison_rows(suite)
    cmp_path = artifacts / "model_comparison.csv"
    with cmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cmp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cmp_rows)
    write_json(artifacts / "model_comparison.json", cmp_rows)

    failures = collect_failures(suite.get("base", {}).get("predictions") or [])
    write_json(artifacts / "error_examples.json", failures)

    gpu_run = str(cfg.get("profile", "")).startswith("cuda") or artifacts.name == "gpu"
    comparison_doc = docs / ("GPU_MODEL_COMPARISON.md" if gpu_run else "MODEL_COMPARISON.md")
    if gpu_run:
        header = (
            "# GPU model comparison\n\n"
            "Locked test split. No hyperparameter tuning on test. "
            "Values copied from `artifacts/gpu/model_comparison.csv`. "
            "Do not paste these into the Mac table in README.md.\n\n"
        )
        footer = "\n"
    else:
        header = (
            "# Model comparison\n\n"
            "Locked test split. No hyperparameter tuning on test. "
            "Values copied from `artifacts/model_comparison.csv`.\n\n"
        )
        footer = "\nProduction CUDA Qwen3-4B numbers: **Not benchmarked on this hardware.**\n"
    comparison_doc.write_text(header + _md_table(cmp_rows) + footer, encoding="utf-8")
    fail_lines = ["# Error analysis\n", "Examples from the evaluated run (base backend).\n"]
    for item in failures:
        fail_lines.append(
            f"- `{item['incident_id']}` gold=`{item['gold']}` pred=`{item['pred']}` error=`{item.get('error')}`"
        )
    if not failures:
        fail_lines.append("- No root-cause mismatches in the evaluated sample.")
    error_doc = docs / ("GPU_ERROR_ANALYSIS.md" if gpu_run else "ERROR_ANALYSIS.md")
    error_doc.write_text("\n".join(fail_lines) + "\n", encoding="utf-8")
    print(json.dumps(cmp_rows, indent=2, default=str))


if __name__ == "__main__":
    main()
