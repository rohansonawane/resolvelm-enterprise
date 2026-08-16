"""Sklearn fallback trainer used when Transformers/TRL cannot run on this host."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

from resolvelm.utils.io import read_jsonl, write_json


def _texts_and_labels(path: Path, text_key: str, label_key: str) -> tuple[list[str], list[str]]:
    rows = read_jsonl(path)
    texts: list[str] = []
    labels: list[str] = []
    for row in rows:
        if text_key == "messages":
            texts.append(row.get("text") or str(row.get("messages")))
        elif text_key == "prompt":
            texts.append(row.get("prompt") + "\n" + row.get("chosen", ""))
        else:
            texts.append(str(row.get(text_key, "")))
        labels.append(str(row[label_key]))
    return texts, labels


def train_classifier_sft(
    train_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    max_samples: int | None = None,
    name: str = "sft",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    texts, labels = _texts_and_labels(train_path, "messages", "root_cause_code")
    sev_texts, severities = _texts_and_labels(train_path, "messages", "severity")
    if max_samples:
        texts, labels = texts[:max_samples], labels[:max_samples]
        sev_texts, severities = sev_texts[:max_samples], severities[:max_samples]
    vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=1)
    x = vectorizer.fit_transform(texts)
    clf = SGDClassifier(loss="log_loss", random_state=seed, max_iter=8, tol=None)
    curve: list[dict[str, float]] = []
    classes = np.unique(labels)
    for epoch in range(1, 4):
        clf.partial_fit(x, labels, classes=classes)
        proba = clf.predict_proba(x)
        y_idx = np.array([list(clf.classes_).index(y) for y in labels])
        loss = float(-np.mean(np.log(np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-9, 1))))
        acc = float(np.mean(clf.predict(x) == np.array(labels)))
        curve.append({"epoch": epoch, "train_loss": loss, "train_accuracy": acc})
        if np.isnan(loss):
            raise RuntimeError("NaN loss in fallback SFT trainer")
    sev = LogisticRegression(max_iter=200, random_state=seed)
    sev.fit(x, severities)
    artifact = {
        "model": clf,
        "vectorizer": vectorizer,
        "severity_model": sev,
        "version": f"fallback-sft-{seed}",
        "name": name,
    }
    joblib.dump(artifact, output_dir / "classifier.joblib")
    summary = {
        "backend": "fallback_sgd",
        "n_train": len(texts),
        "loss_curve": curve,
        "final_train_loss": curve[-1]["train_loss"],
        "final_train_accuracy": curve[-1]["train_accuracy"],
        "adapter_equivalent": str(output_dir / "classifier.joblib"),
        "executed": True,
        "transformers_used": False,
    }
    write_json(output_dir / "train_metrics.json", summary)
    write_json(output_dir / "loss_curve.json", curve)
    return summary


def train_classifier_dpo(
    train_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    max_samples: int | None = None,
    name: str = "dpo",
) -> dict[str, Any]:
    """Train a preference-aware classifier: chosen JSON is positive, rejected is negative."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(train_path)
    if max_samples:
        rows = rows[:max_samples]
    texts: list[str] = []
    labels: list[int] = []
    cause_texts: list[str] = []
    cause_labels: list[str] = []
    for row in rows:
        prompt = row["prompt"]
        texts.append(prompt + "\n" + row["chosen"])
        labels.append(1)
        texts.append(prompt + "\n" + row["rejected"])
        labels.append(0)
        cause_texts.append(prompt)
        cause_labels.append(str(row["root_cause_code"]))
    vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
    x = vectorizer.fit_transform(texts)
    pref = SGDClassifier(loss="log_loss", random_state=seed, max_iter=6, tol=None)
    curve: list[dict[str, float]] = []
    classes = np.array([0, 1])
    for epoch in range(1, 4):
        pref.partial_fit(x, labels, classes=classes)
        proba = pref.predict_proba(x)[:, 1]
        y = np.array(labels)
        loss = float(
            -np.mean(
                y * np.log(np.clip(proba, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - proba, 1e-9, 1))
            )
        )
        acc = float(np.mean((proba >= 0.5) == y))
        curve.append({"epoch": epoch, "train_loss": loss, "preference_accuracy": acc})
        if np.isnan(loss):
            raise RuntimeError("NaN loss in fallback DPO trainer")
    cause_vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
    xc = cause_vec.fit_transform(cause_texts)
    cause_clf = LogisticRegression(max_iter=200, random_state=seed)
    cause_clf.fit(xc, cause_labels)
    artifact = {
        "model": cause_clf,
        "vectorizer": cause_vec,
        "preference_model": pref,
        "preference_vectorizer": vectorizer,
        "severity_model": None,
        "version": f"fallback-dpo-{seed}",
        "name": name,
    }
    joblib.dump(artifact, output_dir / "classifier.joblib")
    summary = {
        "backend": "fallback_preference_sgd",
        "n_pairs": len(rows),
        "loss_curve": curve,
        "final_train_loss": curve[-1]["train_loss"],
        "final_preference_accuracy": curve[-1]["preference_accuracy"],
        "adapter_equivalent": str(output_dir / "classifier.joblib"),
        "executed": True,
        "transformers_used": False,
    }
    write_json(output_dir / "train_metrics.json", summary)
    write_json(output_dir / "loss_curve.json", curve)
    return summary
