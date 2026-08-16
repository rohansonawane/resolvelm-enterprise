# Data Quality Report

Generated from `artifacts/data_quality_report.json`. Numbers are measured, not invented.

- Records: **48**
- Issues: **0**
- Passed hard checks: **True**
- Split counts: `{"train": 32, "val": 6, "test": 10}`
- Label origins: `{"deterministic_template": 24, "derived": 24}`
- Root-cause counts: `{"DISK_FULL": 4, "HDFS_BLOCK_MISSING": 4, "NETWORK_PARTITION": 4, "NODE_FAILURE": 4, "YARN_CONTAINER_OOM": 4, "DB_CONNECTION_POOL_EXHAUSTION": 4, "SERVICE_TIMEOUT": 4, "OPENSTACK_NOVA_SCHEDULER_FAILURE": 4, "MEMORY_PRESSURE": 4, "PERMISSION_DENIED": 4, "CONFIG_MISMATCH": 4, "AUTH_TOKEN_EXPIRED": 4}`
- Family split leakage groups: **0**
- Near-duplicates across split: **0**

## Leakage prevention

- Deduplicate exact bodies before split.
- Group by `family_id` so variants of the same incident stay in one split.
- Sort families by earliest timestamp (grouped temporal split).
- DPO pairs are built from train/val only; test answers are never used.

## Issue type histogram

```json
{}
```
