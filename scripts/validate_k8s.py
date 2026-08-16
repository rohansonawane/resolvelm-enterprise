#!/usr/bin/env python3
"""Validate Kubernetes/KServe YAML without needing a live cluster."""

from __future__ import annotations

import yaml

from resolvelm.utils.config import project_root


def main() -> None:
    root = project_root() / "deploy"
    errors = []
    for path in sorted(root.rglob("*.yaml")):
        try:
            docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
            if not docs:
                errors.append(f"{path}: empty")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc}")
        else:
            print(f"ok {path.relative_to(project_root())} ({len(docs)} docs)")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
