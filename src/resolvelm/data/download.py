"""Attempt to fetch Loghub sample files; always able to fall back to local fixtures."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

LOGHUB_SAMPLES = {
    "HDFS": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log",
    "Hadoop": "https://raw.githubusercontent.com/logpai/loghub/master/Hadoop/Hadoop_2k.log",
    "OpenStack": "https://raw.githubusercontent.com/logpai/loghub/master/OpenStack/openstack_2k.log",
}


def try_download_loghub(out_dir: Path, timeout_s: int = 20) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"attempted": True, "downloaded": {}, "errors": {}}
    for name, url in LOGHUB_SAMPLES.items():
        dest = out_dir / f"{name.lower()}_sample.log"
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as response:
                content = response.read()
            if len(content) < 200:
                raise ValueError("payload too small to be a Loghub sample")
            dest.write_bytes(content)
            result["downloaded"][name] = {
                "path": str(dest),
                "bytes": len(content),
                "label_origin": "original",
            }
        except Exception as exc:  # noqa: BLE001 — network is optional
            result["errors"][name] = str(exc)
    manifest = out_dir / "download_manifest.json"
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
