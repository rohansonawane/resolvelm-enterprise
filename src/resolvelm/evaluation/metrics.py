from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_report_safe(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "labels": []}
    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = {
        label: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i, label in enumerate(labels)
    }
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    return {
        "accuracy": acc,
        "macro_f1": macro,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "labels": labels,
        "sklearn_report": report,
        "pred_counts": dict(Counter(y_pred)),
        "true_counts": dict(Counter(y_true)),
    }


def latency_stats(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "n": 0}
    arr = np.array(values_ms, dtype=float)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(np.mean(arr)),
        "n": int(len(arr)),
    }


def set_prf(pred: set[str], gold: set[str]) -> dict[str, float]:
    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
