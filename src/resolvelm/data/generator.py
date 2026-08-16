"""Deterministic incident generator inspired by Loghub Hadoop/HDFS/OpenStack failure patterns."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from resolvelm.schemas.analysis import RecommendedAction, Severity
from resolvelm.schemas.incident import (
    IncidentRecord,
    LabelOrigin,
    LogEvent,
    MetricEvent,
    TraceEvent,
)

ROOT_CAUSES: dict[str, dict[str, Any]] = {
    "DISK_FULL": {
        "text": "Local disk or HDFS datanode volume is full",
        "severity": Severity.SEV_2,
        "service": "hdfs-datanode",
        "component": "datanode",
        "source": "hdfs",
        "actions": [
            ("expand_volume", "Add capacity or drain the full volume"),
            ("purge_tmp", "Clear stale shuffle and tmp files after confirming they are unused"),
        ],
    },
    "HDFS_BLOCK_MISSING": {
        "text": "HDFS reported missing or under-replicated blocks",
        "severity": Severity.SEV_2,
        "service": "hdfs-namenode",
        "component": "namenode",
        "source": "hdfs",
        "actions": [
            ("fsck_blocks", "Run HDFS fsck on the affected path and restore missing replicas"),
            ("check_datanodes", "Verify datanode liveness and disk health"),
        ],
    },
    "NETWORK_PARTITION": {
        "text": "Network partition or timeout between cluster nodes",
        "severity": Severity.SEV_2,
        "service": "hadoop-yarn",
        "component": "nodemanager",
        "source": "hadoop",
        "actions": [
            ("check_network_path", "Trace connectivity between the reporting nodes"),
            ("review_firewall", "Confirm security group and MTU settings"),
        ],
    },
    "NODE_FAILURE": {
        "text": "Worker node lost heartbeats and was marked lost",
        "severity": Severity.SEV_1,
        "service": "hadoop-yarn",
        "component": "resourcemanager",
        "source": "hadoop",
        "actions": [
            ("fence_node", "Cordon the lost node and reschedule containers"),
            ("inspect_hardware", "Check BMC/console for hardware faults"),
        ],
    },
    "YARN_CONTAINER_OOM": {
        "text": "YARN container killed because it exceeded memory limits",
        "severity": Severity.SEV_3,
        "service": "hadoop-mapreduce",
        "component": "mrappmaster",
        "source": "hadoop",
        "actions": [
            ("raise_container_mb", "Increase mapreduce.map.memory.mb after profiling"),
            ("fix_spill", "Reduce in-memory buffer pressure and spill earlier"),
        ],
    },
    "DB_CONNECTION_POOL_EXHAUSTION": {
        "text": "Database connection pool exhaustion",
        "severity": Severity.SEV_2,
        "service": "nova-api",
        "component": "nova-conductor",
        "source": "openstack",
        "actions": [
            (
                "inspect_connection_leaks",
                "Inspect ORM sessions and connection leaks in nova-conductor",
            ),
            ("raise_pool_ceiling", "Temporarily raise max_pool_size only after leak review"),
        ],
    },
    "SERVICE_TIMEOUT": {
        "text": "Upstream RPC or HTTP deadline exceeded",
        "severity": Severity.SEV_3,
        "service": "nova-compute",
        "component": "nova-compute",
        "source": "openstack",
        "actions": [
            ("trace_rpc", "Trace the slow RPC from compute to conductor"),
            ("scale_workers", "Add workers if queue depth is saturated"),
        ],
    },
    "OPENSTACK_NOVA_SCHEDULER_FAILURE": {
        "text": "Nova scheduler could not place an instance",
        "severity": Severity.SEV_2,
        "service": "nova-scheduler",
        "component": "filter_scheduler",
        "source": "openstack",
        "actions": [
            ("check_host_aggregates", "Review host aggregates, AZ capacity, and filter logs"),
            ("free_capacity", "Reclaim unused instances or add compute hosts"),
        ],
    },
    "MEMORY_PRESSURE": {
        "text": "Host memory pressure causing allocation failures",
        "severity": Severity.SEV_2,
        "service": "hdfs-datanode",
        "component": "jvm",
        "source": "hdfs",
        "actions": [
            ("heap_dump", "Capture a heap dump if GC time is elevated"),
            ("tune_xmx", "Adjust JVM heap only after confirming native memory is not the cause"),
        ],
    },
    "PERMISSION_DENIED": {
        "text": "Process lacked filesystem or Kerberos permissions",
        "severity": Severity.SEV_3,
        "service": "hadoop-hdfs",
        "component": "dfsclient",
        "source": "hadoop",
        "actions": [
            ("audit_acl", "Audit HDFS ACLs and Kerberos principal mapping"),
            ("renew_ticket", "Renew the service keytab ticket"),
        ],
    },
    "CONFIG_MISMATCH": {
        "text": "Incompatible configuration across cluster roles",
        "severity": Severity.SEV_3,
        "service": "hadoop-common",
        "component": "configuration",
        "source": "hadoop",
        "actions": [
            ("diff_site_xml", "Diff core-site and yarn-site against the golden config"),
            ("roll_restart", "Roll a controlled restart after config alignment"),
        ],
    },
    "AUTH_TOKEN_EXPIRED": {
        "text": "Keystone or Kerberos token expired mid-request",
        "severity": Severity.SEV_3,
        "service": "keystone",
        "component": "token",
        "source": "openstack",
        "actions": [
            ("refresh_token", "Refresh the service user token and check clock skew"),
            ("extend_ttl", "Review token TTL only if refresh storms are confirmed"),
        ],
    },
}

ENVIRONMENTS = ("prod", "staging", "dr")
REJECTION_CATEGORIES = (
    "wrong_root_cause",
    "unsupported_evidence",
    "invalid_json",
    "missing_evidence",
    "dangerous_remediation",
    "vague_response",
    "severity_error",
    "excessive_certainty",
)

SYSTEM_PROMPT = (
    "You are ResolveLM, an on-call incident analyst. Return ONLY a JSON object matching "
    "the IncidentAnalysis schema. Cite evidence source_id values that appear in the provided "
    "logs/metrics/traces. Do not invent telemetry. If evidence is weak, set needs_human_review "
    "true and lower confidence. Never recommend destructive actions without corroborating evidence."
)

CODES = tuple(ROOT_CAUSES.keys())


def _ts(index: int, variant: int) -> datetime:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    return base + timedelta(hours=index * 3 + variant, minutes=variant * 7)


def log_lines(code: str, variant: int, host: str) -> list[tuple[str, str, str]]:
    table: dict[str, list[tuple[str, str, str]]] = {
        "DISK_FULL": [
            ("WARN", "datanode", f"No space left on device for volume /data/{variant} on {host}"),
            ("ERROR", "datanode", "DataXceiver: writeBlock failed: No space left on device"),
            ("INFO", "datanode", f"FSDataset volume /data/{variant} usage=99%"),
        ],
        "HDFS_BLOCK_MISSING": [
            ("ERROR", "namenode", f"Missing replica for blk_{900000 + variant} on {host}"),
            ("WARN", "namenode", "Under-replicated blocks remaining=14"),
            ("INFO", "namenode", "BlockManager: scheduled replication for missing blocks"),
        ],
        "NETWORK_PARTITION": [
            (
                "ERROR",
                "nodemanager",
                f"Lost contact with ResourceManager after 15000ms from {host}",
            ),
            ("WARN", "nodemanager", "java.net.SocketTimeoutException: connect timed out"),
            ("INFO", "nodemanager", "Retrying heartbeat to rm1.cluster.internal"),
        ],
        "NODE_FAILURE": [
            ("ERROR", "resourcemanager", f"Node {host} is lost. 3 containers will be rescheduled"),
            ("WARN", "resourcemanager", f"NM heartbeat expired for {host}"),
            ("INFO", "resourcemanager", "Transitioned node to LOST"),
        ],
        "YARN_CONTAINER_OOM": [
            ("ERROR", "mrappmaster", f"Container killed on request. Exit code is 137 on {host}"),
            ("WARN", "nodemanager", "ProcessTree: memory used 4352MB > limit 4096MB"),
            ("INFO", "mrappmaster", "Task attempt failed due to OOM, retrying"),
        ],
        "DB_CONNECTION_POOL_EXHAUSTION": [
            (
                "ERROR",
                "nova-conductor",
                "TimeoutError: QueuePool limit of size 15 overflow 10 reached",
            ),
            ("WARN", "nova-conductor", "Connection pool reported exhaustion; waiting 30s"),
            ("INFO", "nova-api", f"Instance action delayed for req-{variant} due to DB wait"),
        ],
        "SERVICE_TIMEOUT": [
            (
                "ERROR",
                "nova-compute",
                "MessagingTimeout: RPC timeout calling conductor.build_and_run",
            ),
            ("WARN", "nova-compute", f"Instance spawn exceeded 90s deadline on {host}"),
            ("INFO", "nova-compute", "Retrying claim resources"),
        ],
        "OPENSTACK_NOVA_SCHEDULER_FAILURE": [
            (
                "ERROR",
                "filter_scheduler",
                "NoValidHost: No valid host was found. There are not enough hosts available",
            ),
            ("WARN", "filter_scheduler", "RamFilter failed on 12 hosts; DiskFilter failed on 3"),
            ("INFO", "nova-scheduler", f"Select destinations failed for request req-{variant}"),
        ],
        "MEMORY_PRESSURE": [
            ("WARN", "jvm", f"GC overhead limit approaching on {host}: pause=1840ms"),
            ("ERROR", "datanode", "java.lang.OutOfMemoryError: Java heap space"),
            ("INFO", "datanode", "Heap used 98% of 4g"),
        ],
        "PERMISSION_DENIED": [
            (
                "ERROR",
                "dfsclient",
                f"org.apache.hadoop.security.AccessControlException: Permission denied: user=etl, path=/data/job-{variant}",
            ),
            ("WARN", "namenode", "Audit: denied open on /data for etl"),
            ("INFO", "dfsclient", "Kerberos principal etl/jobs@CLUSTER.LOCAL"),
        ],
        "CONFIG_MISMATCH": [
            ("ERROR", "configuration", "IPC version mismatch: client protocol 9 vs server 8"),
            (
                "WARN",
                "nodemanager",
                "yarn.nodemanager.resource.memory-mb differs from RM expected value",
            ),
            ("INFO", "nodemanager", f"Loaded yarn-site.xml from /etc/hadoop/{host}"),
        ],
        "AUTH_TOKEN_EXPIRED": [
            (
                "ERROR",
                "token",
                "Unauthorized: The request you have made requires authentication (token expired)",
            ),
            ("WARN", "keystone", f"Token expired at 2024-03-02T0{variant % 9}:00:00Z"),
            ("INFO", "nova-api", "Rejecting request with expired Keystone token"),
        ],
    }
    return table[code]


def gold_analysis(record: IncidentRecord) -> dict[str, Any]:
    evidence = [
        {"source_id": eid, "reason": f"Event {eid} supports {record.root_cause_code}"}
        for eid in record.evidence_ids
    ]
    return {
        "severity": record.severity.value,
        "root_cause_code": record.root_cause_code,
        "root_cause": record.root_cause_text,
        "confidence": 0.86,
        "evidence": evidence,
        "recommended_actions": [a.model_dump() for a in record.recommended_actions],
        "needs_human_review": record.severity in {Severity.SEV_1, Severity.SEV_2},
    }


def _build_incident(index: int, variant: int, code: str) -> IncidentRecord:
    meta = ROOT_CAUSES[code]
    family_id = f"FAM-{code}-{index:03d}"
    incident_id = f"INC-{code}-{index:03d}-v{variant}"
    host = f"node-{index % 17}-{variant}.cluster.internal"
    ts = _ts(index, variant)
    env = ENVIRONMENTS[index % len(ENVIRONMENTS)]
    lines = log_lines(code, variant, host)
    logs = []
    for i, (level, component, message) in enumerate(lines, start=1):
        logs.append(
            LogEvent(
                event_id=f"log-{i}",
                timestamp=ts + timedelta(seconds=i * 3),
                level=level,
                component=component,
                message=message,
            )
        )
    metrics = [
        MetricEvent(
            event_id="metric-1",
            timestamp=ts,
            name="error_rate",
            value=float(5 + variant),
            unit="count/min",
        )
    ]
    traces = [
        TraceEvent(
            event_id="trace-1",
            timestamp=ts,
            span=f"{meta['service']}.handle",
            duration_ms=120.0 + variant * 15,
            status="error" if meta["severity"] in {Severity.SEV_1, Severity.SEV_2} else "ok",
        )
    ]
    error_ids = [e.event_id for e in logs if e.level in {"ERROR", "WARN"}]
    actions = [
        RecommendedAction(action_id=aid, priority=i, description=desc)
        for i, (aid, desc) in enumerate(meta["actions"], start=1)
    ]
    origin = LabelOrigin.DETERMINISTIC_TEMPLATE if variant == 0 else LabelOrigin.DERIVED
    symptoms = [
        f"{meta['text']} observed on {host}",
        f"{meta['component']} emitting {lines[0][0]} in {env}",
    ]
    return IncidentRecord(
        incident_id=incident_id,
        family_id=family_id,
        source_dataset=str(meta["source"]),
        split="unsplit",
        timestamp=ts,
        service=str(meta["service"]),
        component=str(meta["component"]),
        environment=env,
        symptoms=symptoms,
        log_events=logs,
        metric_events=metrics,
        trace_events=traces,
        root_cause_code=code,
        root_cause_text=str(meta["text"]),
        severity=meta["severity"],
        evidence_ids=error_ids[:2],
        action_ids=[a.action_id for a in actions],
        recommended_actions=actions,
        label_origin=origin,
        extra={"host": host, "variant": variant},
    )


def generate_corpus(
    n_families: int,
    n_variants_per_family: int,
    seed: int = 42,
) -> list[IncidentRecord]:
    records: list[IncidentRecord] = []
    for i in range(n_families):
        code = CODES[i % len(CODES)]
        for variant in range(n_variants_per_family):
            records.append(_build_incident(index=i, variant=variant, code=code))
    return records


def render_user_prompt(payload: dict[str, Any]) -> str:
    logs = payload.get("log_events") or []
    log_lines_txt = "\n".join(
        f"- [{item.get('event_id')}] {item.get('level')} {item.get('component')}: {item.get('message')}"
        for item in logs
    )
    symptoms = "; ".join(payload.get("symptoms") or [])
    return (
        f"Service: {payload.get('service')}\n"
        f"Environment: {payload.get('environment')}\n"
        f"Symptoms: {symptoms}\n"
        f"Logs:\n{log_lines_txt or '- none'}\n"
        "Return strict JSON only."
    )
