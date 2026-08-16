# Dataset card — ResolveLM incident corpus

## Motivation

There is no redistributed proprietary enterprise ticket dump in this repo. The training corpus is a **documented derived/synthetic** set inspired by public Loghub Hadoop, HDFS, and OpenStack failure *patterns* (disk full, missing HDFS blocks, YARN OOM, Nova pool exhaustion, scheduler `NoValidHost`, token expiry, and related ops failures).

`scripts/download_data.py` attempts to fetch Loghub 2k-line samples when the network allows. Those files, if present, are **original logs used as style references**. Labels in the JSONL corpus remain `derived` or `deterministic_template`. Preference pairs are `synthetic_preference`.

## Fields

`incident_id`, `family_id`, `source_dataset`, `split`, `timestamp`, `service`, `component`, `environment`, `symptoms[]`, `log_events[]`, `metric_events[]`, `trace_events[]`, `root_cause_code`, `root_cause_text`, `severity`, `evidence_ids[]`, `action_ids[]`, `recommended_actions[]`, `label_origin`.

`label_origin ∈ {original, derived, deterministic_template, synthetic_preference}`.

## Splits

Grouped temporal split by `family_id`. Variants of the same incident stay in one split. See `docs/DATA_QUALITY_REPORT.md` (generated) and `artifacts/data_quality_report.json`.

## Taxonomy (root_cause_code)

DISK_FULL, HDFS_BLOCK_MISSING, NETWORK_PARTITION, NODE_FAILURE, YARN_CONTAINER_OOM, DB_CONNECTION_POOL_EXHAUSTION, SERVICE_TIMEOUT, OPENSTACK_NOVA_SCHEDULER_FAILURE, MEMORY_PRESSURE, PERMISSION_DENIED, CONFIG_MISMATCH, AUTH_TOKEN_EXPIRED.

## Limitations

- Not production pager traffic.
- Remediation text is templated, not copied from a real runbook system of record.
- Near-duplicate n-gram overlap across same-pattern families is reported in the quality JSON; grouped split still blocks family leakage.
