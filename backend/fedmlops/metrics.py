"""Evaluation metrics, implemented in NumPy.

Kept dependency-free on purpose: the demo should run with numpy + pandas only.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), ties handled with average ranks."""
    y_true = np.asarray(y_true).astype(int)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

    # average ranks over tied score groups
    sorted_scores = np.asarray(y_score)[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1

    sum_pos_ranks = float(np.sum(ranks[y_true == 1]))
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion(y_true, y_pred)

    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": round(accuracy(y_true, y_pred), 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "roc_auc": round(roc_auc(y_true, y_prob), 6),
        "log_loss": round(
            float(
                -np.mean(
                    y_true * np.log(np.clip(y_prob, 1e-12, 1))
                    + (1 - y_true) * np.log(np.clip(1 - y_prob, 1e-12, 1))
                )
            ),
            6,
        ),
        "n": int(len(y_true)),
    }
