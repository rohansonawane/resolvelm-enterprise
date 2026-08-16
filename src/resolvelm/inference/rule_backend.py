"""Keyword / pattern baseline. Deterministic, not a silent fake of an LLM."""

from __future__ import annotations

import json
import re

from resolvelm.data.generator import ROOT_CAUSES, SYSTEM_PROMPT
from resolvelm.schemas.analysis import IncidentAnalysis


def _context_ids(prompt: str) -> list[str]:
    return re.findall(r"\[(log-\d+|metric-\d+|trace-\d+)\]", prompt)


RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "DB_CONNECTION_POOL_EXHAUSTION",
        ("queuepool", "connection pool", "pool exhaustion", "overflow"),
    ),
    ("DISK_FULL", ("no space left", "usage=99%", "disk full")),
    ("HDFS_BLOCK_MISSING", ("missing replica", "under-replicated", "blk_")),
    ("NETWORK_PARTITION", ("sockettimeoutexception", "lost contact", "connect timed out")),
    ("NODE_FAILURE", ("node", "is lost", "heartbeat expired", "transitioned node to lost")),
    ("YARN_CONTAINER_OOM", ("exit code is 137", "memory used", "> limit")),
    ("OPENSTACK_NOVA_SCHEDULER_FAILURE", ("novalidhost", "not enough hosts", "ramfilter")),
    ("SERVICE_TIMEOUT", ("messagingtimeout", "rpc timeout", "deadline")),
    ("MEMORY_PRESSURE", ("outofmemoryerror", "java heap space", "gc overhead")),
    ("PERMISSION_DENIED", ("accesscontrolexception", "permission denied")),
    ("CONFIG_MISMATCH", ("ipc version mismatch", "yarn-site", "differs from rm")),
    ("AUTH_TOKEN_EXPIRED", ("token expired", "requires authentication")),
]


class RuleBasedBackend:
    name = "rule"
    version = "rule-baseline-v1"

    def __init__(self) -> None:
        self.system_prompt = SYSTEM_PROMPT

    def _match(self, prompt: str) -> str:
        blob = prompt.lower()
        scores: list[tuple[int, str]] = []
        for code, needles in RULES:
            hit = sum(1 for n in needles if n in blob)
            if hit:
                scores.append((hit, code))
        if not scores:
            return "SERVICE_TIMEOUT"
        scores.sort(reverse=True)
        return scores[0][1]

    def generate_sync(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        del max_new_tokens
        code = self._match(prompt)
        meta = ROOT_CAUSES[code]
        source_ids = _context_ids(prompt) or ["log-1"]
        evidence = [
            {"source_id": source_ids[0], "reason": f"Pattern match for {code}"},
        ]
        if len(source_ids) > 1:
            evidence.append({"source_id": source_ids[1], "reason": "Secondary corroborating event"})
        actions = [
            {
                "action_id": aid,
                "priority": i,
                "description": desc,
            }
            for i, (aid, desc) in enumerate(meta["actions"], start=1)
        ]
        payload = {
            "severity": meta["severity"].value,
            "root_cause_code": code,
            "root_cause": meta["text"],
            "confidence": 0.74,
            "evidence": evidence,
            "recommended_actions": actions,
            "needs_human_review": True,
        }
        IncidentAnalysis.model_validate(payload)
        return json.dumps(payload, separators=(",", ":"))

    async def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        return self.generate_sync(prompt, max_new_tokens=max_new_tokens)
