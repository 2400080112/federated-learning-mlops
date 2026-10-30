"""ORM objects -> API payloads. Kept in one place so list and detail stay consistent."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..models import ModelVersion, Run
from ..schemas import ModelVersionOut, RoundPoint, RunDetail, RunSummary


def _config(run: Run) -> Dict[str, Any]:
    try:
        return json.loads(run.config_json or "{}")
    except json.JSONDecodeError:
        return {}


def run_to_summary(run: Run) -> RunSummary:
    cfg = _config(run).get("federated", {})
    return RunSummary(
        id=run.id,
        name=run.name,
        status=run.status,
        mode=run.mode,
        created_at=run.created_at,
        finished_at=run.finished_at,
        duration_s=run.duration_s,
        best_round=run.best_round,
        rounds_run=run.rounds_run,
        n_clients=cfg.get("n_clients"),
        partition=cfg.get("partition"),
        privacy_enabled=run.privacy_enabled,
        epsilon=run.epsilon,
        val_accuracy=run.val_accuracy,
        test_accuracy=run.test_accuracy,
        test_f1=run.test_f1,
        test_roc_auc=run.test_roc_auc,
        is_scored=run.is_scored,
        error=run.error,
    )


def run_to_detail(run: Run) -> RunDetail:
    base = run_to_summary(run).model_dump()
    history = [
        RoundPoint(
            round=m.round_no,
            train_loss=m.train_loss,
            val_accuracy=m.val_accuracy,
            val_f1=m.val_f1,
            val_roc_auc=m.val_roc_auc,
            val_log_loss=m.val_log_loss,
            mean_update_norm=m.mean_update_norm,
            clipped_fraction=m.clipped_fraction,
            epsilon=m.epsilon,
            n_clients=m.n_clients,
        )
        for m in sorted(run.metrics, key=lambda m: m.round_no)
    ]
    test = {}
    if run.test_accuracy is not None:
        test = {
            "accuracy": run.test_accuracy,
            "precision": run.test_precision or 0.0,
            "recall": run.test_recall or 0.0,
            "f1": run.test_f1 or 0.0,
            "roc_auc": run.test_roc_auc or 0.0,
            "log_loss": run.test_log_loss or 0.0,
        }
    return RunDetail(
        **base,
        config=_config(run),
        clients=json.loads(run.clients_json or "[]"),
        history=history,
        test_metrics=test,
        noise_multiplier=run.noise_multiplier,
        clip_norm=run.clip_norm,
        registered_versions=[f"{mv.name}:{mv.version}" for mv in run.model_versions],
    )


def model_to_out(mv: ModelVersion) -> ModelVersionOut:
    try:
        meta = json.loads(mv.metadata_json or "{}")
    except json.JSONDecodeError:
        meta = {}
    return ModelVersionOut(
        id=mv.id,
        name=mv.name,
        version=mv.version,
        stage=mv.stage,
        created_at=mv.created_at,
        updated_at=mv.updated_at,
        run_id=mv.run_id,
        description=mv.description,
        test_accuracy=mv.test_accuracy,
        test_f1=mv.test_f1,
        test_roc_auc=mv.test_roc_auc,
        epsilon=mv.epsilon,
        metadata=meta,
    )
