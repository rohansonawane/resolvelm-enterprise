#!/usr/bin/env python3
"""Download Loghub samples when network/disk allow; always leave a fixture note."""

from __future__ import annotations

import argparse
import json

from resolvelm.data.download import try_download_loghub
from resolvelm.utils.config import project_root
from resolvelm.utils.device import free_disk_gb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--allow-synthetic", action="store_true", default=True)
    args = parser.parse_args()
    root = project_root()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    disk = free_disk_gb()
    result = {"free_disk_gb": disk, "loghub": None, "synthetic_fallback": True}
    if disk >= 1.0:
        result["loghub"] = try_download_loghub(out / "loghub")
        result["synthetic_fallback"] = not bool(result["loghub"].get("downloaded"))
    else:
        result["loghub"] = {"attempted": False, "reason": "insufficient disk"}
    note = {
        "statement": (
            "ResolveLM does not ship a proprietary enterprise ticket dump. "
            "Incidents are derived/synthetic templates inspired by public Loghub Hadoop/HDFS/OpenStack "
            "failure patterns. If Loghub samples downloaded, they are original logs used only as style "
            "references — labels remain derived."
        ),
        "label_origins": [
            "original",
            "derived",
            "deterministic_template",
            "synthetic_preference",
        ],
    }
    (out / "SOURCE.md").write_text(note["statement"] + "\n", encoding="utf-8")
    (out / "download_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
