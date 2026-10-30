"""Scoring a finished run on the held-out test split, and the registry rules.

Two invariants this module enforces:

1. A run cannot be registered until it has a measured test score. "Unmeasured
   models do not ship" is the whole reason a registry exists.
2. Promotion is a pointer move, not a copy: the artifact never changes, so
   rolling back is promoting the previous version again.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from fedmlops import metrics as M
from fedmlops.config import RunConfig
from fedmlops.data import load_splits
from fedmlops.model import LogisticRegression

from ..models import ModelCounter, ModelVersion, Run


class ServiceError(Exception):
    """A rule was violated; the router turns this into a 4xx with the message."""


# ------------------------------------------------------------------ evaluate
def evaluate_run(session: Session, run: Run) -> Dict[str, float]:
    if run.model_json is None:
        raise ServiceError("this run has no trained model yet (status: %s)" % run.status)

    cfg = RunConfig.from_dict(json.loads(run.config_json))
    _, _, test_ds = load_splits(
        cfg.data.csv_path,
        target_column=cfg.data.target_column,
        drop_columns=tuple(cfg.data.drop_columns),
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        seed=cfg.data.seed,
    )
    model = LogisticRegression.from_dict(json.loads(run.model_json))
    scores = M.evaluate(test_ds.y, model.predict_proba(test_ds.X))

    run.test_accuracy = scores["accuracy"]
    run.test_f1 = scores["f1"]
    run.test_roc_auc = scores["roc_auc"]
    run.test_log_loss = scores["log_loss"]
    run.test_precision = scores["precision"]
    run.test_recall = scores["recall"]
    session.commit()
    return scores


def test_metrics(run: Run) -> Dict[str, float]:
    if run.test_accuracy is None:
        return {}
    return {
        "accuracy": run.test_accuracy,
        "precision": run.test_precision or 0.0,
        "recall": run.test_recall or 0.0,
        "f1": run.test_f1 or 0.0,
        "roc_auc": run.test_roc_auc or 0.0,
        "log_loss": run.test_log_loss or 0.0,
    }


# ------------------------------------------------------------------ register
def register_run(
    session: Session,
    run: Run,
    name: str = "wdbc-logreg",
    stage: str = "staging",
    description: str = "",
) -> ModelVersion:
    if run.model_json is None:
        raise ServiceError(f"run {run.id} has no model to register (status: {run.status})")
    if run.test_accuracy is None:
        raise ServiceError(
            "this run has no test score yet. Evaluate it first -- a model should "
            "not enter the registry unmeasured."
        )
    if stage not in {"staging", "production", "archived"}:
        raise ServiceError(f"unknown stage: {stage}")

    # Take the next number from a per-model counter rather than from the rows
    # that exist right now. Counting broke in two ways: count+1 collided with a
    # surviving row after an older version was deleted (500 on the unique
    # constraint), and max+1 re-issued a deleted version's number, so "v2" could
    # mean two different artifacts over time.
    existing: List[str] = session.scalars(
        select(ModelVersion.version).where(ModelVersion.name == name)
    ).all()
    counter = session.get(ModelCounter, name)
    if counter is None:
        used = {int(v[1:]) for v in existing if v.startswith("v") and v[1:].isdigit()}
        counter = ModelCounter(name=name, last_version=max(used, default=0))
        session.add(counter)
    counter.last_version += 1
    version = f"v{counter.last_version}"

    metadata = {
        "source_run": run.id,
        "run_name": run.name,
        "partition": json.loads(run.config_json or "{}").get("federated", {}).get("partition"),
        "n_clients": json.loads(run.config_json or "{}").get("federated", {}).get("n_clients"),
        "rounds_run": run.rounds_run,
        "best_round": run.best_round,
        "val": {
            "accuracy": run.val_accuracy,
            "f1": run.val_f1,
            "roc_auc": run.val_roc_auc,
        },
        "test": test_metrics(run),
        "privacy": {
            "enabled": run.privacy_enabled,
            "epsilon": run.epsilon,
            "noise_multiplier": run.noise_multiplier,
            "clip_norm": run.clip_norm,
        },
        "config": json.loads(run.config_json or "{}"),
    }

    mv = ModelVersion(
        name=name,
        version=version,
        stage=stage,
        run_id=run.id,
        description=description,
        metadata_json=json.dumps(metadata),
        model_json=run.model_json,
        scaler_json=run.scaler_json or "{}",
        test_accuracy=run.test_accuracy,
        test_f1=run.test_f1,
        test_roc_auc=run.test_roc_auc,
        epsilon=run.epsilon,
    )
    session.add(mv)
    session.commit()

    if stage in {"staging", "production"}:
        promote(session, mv, stage=stage, archive_others=True)
    return mv


# ------------------------------------------------------------------ promote
def promote(
    session: Session,
    version: ModelVersion,
    stage: str = "production",
    min_accuracy: Optional[float] = None,
    archive_others: bool = True,
) -> ModelVersion:
    if stage not in {"staging", "production", "archived"}:
        raise ServiceError(f"unknown stage: {stage}")

    if min_accuracy is not None and stage == "production":
        if version.test_accuracy is None:
            raise ServiceError(
                f"{version.name}:{version.version} has no test accuracy; cannot gate on min_accuracy"
            )
        if version.test_accuracy < min_accuracy:
            raise ServiceError(
                f"promotion blocked: {version.name}:{version.version} test accuracy "
                f"{version.test_accuracy:.4f} < required {min_accuracy:.4f}"
            )

    now = datetime.now(timezone.utc)
    if archive_others and stage in {"staging", "production"}:
        # one version per stage: archive whoever held it
        others = session.scalars(
            select(ModelVersion).where(
                ModelVersion.name == version.name,
                ModelVersion.stage == stage,
                ModelVersion.id != version.id,
            )
        ).all()
        for other in others:
            other.stage = "archived"
            other.updated_at = now

    version.stage = stage
    version.updated_at = now
    session.commit()
    return version


def production_model(session: Session, name: str = "wdbc-logreg") -> Optional[ModelVersion]:
    return session.scalars(
        select(ModelVersion)
        .where(ModelVersion.name == name, ModelVersion.stage == "production")
        .order_by(ModelVersion.id.desc())
    ).first()
