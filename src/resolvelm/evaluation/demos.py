from __future__ import annotations

from datetime import UTC, datetime, timedelta

from resolvelm.schemas.analysis import RecommendedAction, Severity
from resolvelm.schemas.incident import IncidentRecord, LabelOrigin, LogEvent

DEMO_SPECS = (
    {
        "demo_id": "disk_full",
        "title": "Datanode disk full",
        "code": "DISK_FULL",
        "service": "hdfs-datanode",
        "environment": "prod",
        "symptoms": ["Writes failing on /data/0", "Datanode reports no space left"],
        "logs": [
            (
                "log-1",
                "ERROR",
                "datanode",
                "DataXceiver: writeBlock failed: No space left on device",
            ),
            ("log-2", "WARN", "datanode", "FSDataset volume /data/0 usage=99%"),
        ],
    },
    {
        "demo_id": "network_failure",
        "title": "NodeManager network timeout",
        "code": "NETWORK_PARTITION",
        "service": "hadoop-yarn",
        "environment": "prod",
        "symptoms": ["Heartbeats to ResourceManager timing out"],
        "logs": [
            (
                "log-1",
                "ERROR",
                "nodemanager",
                "Lost contact with ResourceManager after 15000ms from node-7-0.cluster.internal",
            ),
            ("log-2", "WARN", "nodemanager", "java.net.SocketTimeoutException: connect timed out"),
        ],
    },
    {
        "demo_id": "node_failure",
        "title": "YARN node lost",
        "code": "NODE_FAILURE",
        "service": "hadoop-yarn",
        "environment": "prod",
        "symptoms": ["Node marked LOST", "Containers being rescheduled"],
        "logs": [
            (
                "log-1",
                "ERROR",
                "resourcemanager",
                "Node node-3-0.cluster.internal is lost. 3 containers will be rescheduled",
            ),
            (
                "log-2",
                "WARN",
                "resourcemanager",
                "NM heartbeat expired for node-3-0.cluster.internal",
            ),
        ],
    },
    {
        "demo_id": "connection_exhaustion",
        "title": "Nova DB pool exhaustion",
        "code": "DB_CONNECTION_POOL_EXHAUSTION",
        "service": "nova-api",
        "environment": "prod",
        "symptoms": ["Instance actions delayed", "SQLAlchemy pool wait"],
        "logs": [
            (
                "log-1",
                "ERROR",
                "nova-conductor",
                "TimeoutError: QueuePool limit of size 15 overflow 10 reached",
            ),
            ("log-2", "WARN", "nova-conductor", "Connection pool reported exhaustion; waiting 30s"),
        ],
    },
    {
        "demo_id": "service_timeout",
        "title": "Nova compute RPC timeout",
        "code": "SERVICE_TIMEOUT",
        "service": "nova-compute",
        "environment": "staging",
        "symptoms": ["Instance spawn exceeded deadline"],
        "logs": [
            (
                "log-1",
                "ERROR",
                "nova-compute",
                "MessagingTimeout: RPC timeout calling conductor.build_and_run",
            ),
            (
                "log-2",
                "WARN",
                "nova-compute",
                "Instance spawn exceeded 90s deadline on node-4-1.cluster.internal",
            ),
        ],
    },
)


def demo_incidents() -> list[dict]:
    out = []
    ts = datetime(2024, 6, 1, tzinfo=UTC)
    for spec in DEMO_SPECS:
        logs = [
            LogEvent(
                event_id=lid,
                timestamp=ts + timedelta(seconds=i),
                level=level,
                component=comp,
                message=msg,
            )
            for i, (lid, level, comp, msg) in enumerate(spec["logs"])
        ]
        rec = IncidentRecord(
            incident_id=f"DEMO-{spec['demo_id']}",
            family_id=f"DEMO-{spec['code']}",
            source_dataset="demo",
            split="test",
            timestamp=ts,
            service=spec["service"],
            component=logs[0].component,
            environment=spec["environment"],
            symptoms=spec["symptoms"],
            log_events=logs,
            metric_events=[],
            trace_events=[],
            root_cause_code=spec["code"],
            root_cause_text=spec["title"],
            severity=Severity.SEV_2,
            evidence_ids=[logs[0].event_id],
            action_ids=["triage"],
            recommended_actions=[
                RecommendedAction(
                    action_id="triage", priority=1, description="Triage using runbook"
                )
            ],
            label_origin=LabelOrigin.DETERMINISTIC_TEMPLATE,
            extra={"demo": True, "title": spec["title"]},
        )
        out.append(
            {
                "demo_id": spec["demo_id"],
                "title": spec["title"],
                "is_demo": True,
                "request": {
                    "service": rec.service,
                    "environment": rec.environment,
                    "symptoms": rec.symptoms,
                    "log_events": [e.model_dump(mode="json") for e in rec.log_events],
                    "demo_id": spec["demo_id"],
                },
            }
        )
    return out
