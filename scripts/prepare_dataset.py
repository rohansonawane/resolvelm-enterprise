#!/usr/bin/env python3
"""Generate, validate, split, and convert the ResolveLM incident corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from resolvelm.data.convert import incidents_to_dpo, incidents_to_sft
from resolvelm.data.generator import generate_corpus
from resolvelm.data.leakage import family_leakage
from resolvelm.data.split import grouped_temporal_split
from resolvelm.data.validate import validate_incidents
from resolvelm.utils.config import load_yaml, project_root
from resolvelm.utils.io import write_json, write_jsonl
from resolvelm.utils.seed import set_global_seed


def _dump_records(path: Path, records) -> None:
    write_jsonl(path, [r.model_dump(mode="json") for r in records])


def render_quality_md(report: dict, dest: Path) -> None:
    lines = [
        "# Data Quality Report",
        "",
        "Generated from `artifacts/data_quality_report.json`. Numbers are measured, not invented.",
        "",
        f"- Records: **{report['n_records']}**",
        f"- Issues: **{report['n_issues']}**",
        f"- Passed hard checks: **{report['passed']}**",
        f"- Split counts: `{json.dumps(report['split_counts'])}`",
        f"- Label origins: `{json.dumps(report['label_origin_counts'])}`",
        f"- Root-cause counts: `{json.dumps(report['root_cause_counts'])}`",
        f"- Family split leakage groups: **{len(report['family_split_leakage'])}**",
        f"- Near-duplicates across split: **{report['near_duplicate_across_split']}**",
        "",
        "## Leakage prevention",
        "",
        "- Deduplicate exact bodies before split.",
        "- Group by `family_id` so variants of the same incident stay in one split.",
        "- Sort families by earliest timestamp (grouped temporal split).",
        "- DPO pairs are built from train/val only; test answers are never used.",
        "",
        "## Issue type histogram",
        "",
        "```json",
        json.dumps(report["issue_types"], indent=2),
        "```",
        "",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = project_root()
    cfg_path = root / args.config
    if args.smoke:
        local = root / "configs/local/data.yaml"
        if local.exists():
            cfg_path = local
    cfg = load_yaml(cfg_path)
    set_global_seed(int(cfg.get("seed", 42)))
    records = generate_corpus(
        n_families=int(cfg.get("n_families", 24)),
        n_variants_per_family=int(cfg.get("n_variants_per_family", 2)),
        seed=int(cfg.get("seed", 42)),
    )
    split_cfg = cfg.get("split") or {}
    split_records = grouped_temporal_split(
        records,
        train_ratio=float(split_cfg.get("train_ratio", 0.7)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
    )
    leak_cfg = cfg.get("leakage") or {}
    report = validate_incidents(
        split_records,
        near_dup_ngram=int(leak_cfg.get("near_dup_ngram", 8)),
        near_dup_jaccard=float(leak_cfg.get("near_dup_jaccard", 0.92)),
    )
    train = [r for r in split_records if r.split == "train"]
    val = [r for r in split_records if r.split == "val"]
    test = [r for r in split_records if r.split == "test"]
    report["family_leakage_train_test"] = family_leakage(train, test)
    report["passed"] = report["passed"] and not report["family_leakage_train_test"]

    processed = root / cfg.get("processed_dir", "data/processed")
    processed.mkdir(parents=True, exist_ok=True)
    _dump_records(processed / "incidents.jsonl", split_records)
    _dump_records(processed / "incidents_train.jsonl", train)
    _dump_records(processed / "incidents_val.jsonl", val)
    _dump_records(processed / "incidents_test.jsonl", test)

    sft_train = incidents_to_sft(train)
    sft_val = incidents_to_sft(val)
    dpo_train = incidents_to_dpo(
        train, pairs_per_record=int(cfg.get("preference_pairs_per_train", 2))
    )
    dpo_val = incidents_to_dpo(val, pairs_per_record=1)
    write_jsonl(processed / "sft_train.jsonl", sft_train)
    write_jsonl(processed / "sft_val.jsonl", sft_val)
    write_jsonl(processed / "dpo_train.jsonl", dpo_train)
    write_jsonl(processed / "dpo_val.jsonl", dpo_val)

    artifacts = root / cfg.get("artifacts_dir", "artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    write_json(artifacts / "data_quality_report.json", report)
    write_json(
        artifacts / "dataset_manifest.json",
        {
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_sft_train": len(sft_train),
            "n_dpo_train": len(dpo_train),
            "seed": cfg.get("seed"),
            "config": str(cfg_path),
        },
    )
    docs = root / cfg.get("docs_dir", "docs")
    docs.mkdir(parents=True, exist_ok=True)
    render_quality_md(report, docs / "DATA_QUALITY_REPORT.md")
    print(
        json.dumps(
            {
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "passed": report["passed"],
                "issues": report["n_issues"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
