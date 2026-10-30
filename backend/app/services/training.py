"""Turning an API request into a training job, and running it off the request thread.

FastAPI must return immediately (a 25-round run is fast here, but a 500-round one
is not), so training happens in a worker thread with its own database session.
The run row is the progress channel: ``queued -> running -> finished``/``failed``,
plus one ``round_metrics`` row per round, which is what the UI polls.
"""

from __future__ import annotations

import json
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from sqlalchemy.orm import Session, sessionmaker

from fedmlops.config import RunConfig
from fedmlops.data import load_splits
from fedmlops.privacy import sigma_for_epsilon
from fedmlops.trainer import RoundRecord, train_federated

from ..models import RoundMetric, Run
from ..schemas import RunCreate

# The pool belongs to the application instance, not to this module: a shared
# global pool leaked work between app instances (and between tests), so one
# app's queued runs could starve another's. See create_app().
MAX_WORKERS = 2


def make_executor() -> ThreadPoolExecutor:
    """Two workers: enough to show concurrent runs, few enough that SQLite and
    the CPU stay honest. A real deployment would hand this to a task queue."""
    return ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="fl-train")


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"run-{stamp}-{uuid.uuid4().hex[:4]}"


def build_config(payload: RunCreate) -> RunConfig:
    """Map the API payload onto the core library's config object."""
    cfg = RunConfig()
    cfg.seed = payload.seed
    cfg.data.seed = payload.seed

    fed = cfg.federated
    fed.n_clients = payload.n_clients
    fed.clients_per_round = payload.effective_clients_per_round
    fed.rounds = payload.rounds
    fed.local_epochs = payload.local_epochs
    fed.batch_size = payload.batch_size
    fed.learning_rate = payload.learning_rate
    fed.l2 = payload.l2
    fed.partition = payload.partition
    fed.dirichlet_alpha = payload.dirichlet_alpha

    priv = cfg.privacy
    priv.enabled = payload.privacy_enabled
    priv.clip_norm = payload.dp_clip
    priv.delta = payload.dp_delta
    if payload.privacy_enabled:
        if payload.dp_sigma is not None:
            priv.noise_multiplier = payload.dp_sigma
        else:
            priv.noise_multiplier = sigma_for_epsilon(
                payload.dp_epsilon, payload.rounds, payload.dp_delta
            )
        priv.max_epsilon = payload.dp_epsilon or 0.0

    cfg.validate()
    return cfg


def create_run(session: Session, payload: RunCreate) -> Run:
    """Insert a queued run row. Training is started separately by the router."""
    cfg = build_config(payload)
    run = Run(
        id=new_run_id(),
        name=payload.name or _default_name(payload),
        status="queued",
        mode="federated",
        config_json=json.dumps(cfg.to_dict()),
        clients_json="[]",
        privacy_enabled=cfg.privacy.enabled,
        epsilon=None,
        noise_multiplier=cfg.privacy.noise_multiplier if cfg.privacy.enabled else None,
        clip_norm=cfg.privacy.clip_norm if cfg.privacy.enabled else None,
    )
    session.add(run)
    session.commit()
    return run


def _default_name(payload: RunCreate) -> str:
    tag = "dp" if payload.privacy_enabled else payload.partition
    return f"{tag} · {payload.n_clients}c · {payload.rounds}r"


def start_training(
    executor: ThreadPoolExecutor, session_factory: sessionmaker[Session], run_id: str
) -> None:
    """Queue a run on this application's worker pool (what the HTTP endpoint does)."""
    executor.submit(run_training, session_factory, run_id)


def run_training(
    session_factory: sessionmaker[Session],
    run_id: str,
    on_round: Optional[Callable[[RoundRecord], None]] = None,
) -> None:
    """Train one queued run to completion, synchronously.

    Called from the worker thread normally, and directly by the seed script and
    the terminal CLI. ``on_round`` lets a caller print progress as rounds land;
    the HTTP path does not need it because the browser polls the database.
    """
    session = session_factory()
    try:
        run = session.get(Run, run_id)
        if run is None:
            return
        raw = json.loads(run.config_json)
        cfg = RunConfig.from_dict(raw)

        run.status = "running"
        session.commit()

        train_ds, val_ds, _ = load_splits(
            cfg.data.csv_path,
            target_column=cfg.data.target_column,
            drop_columns=tuple(cfg.data.drop_columns),
            test_size=cfg.data.test_size,
            val_size=cfg.data.val_size,
            seed=cfg.data.seed,
        )

        def persist_round(record: RoundRecord) -> None:
            session.add(
                RoundMetric(
                    run_id=run_id,
                    round_no=record.round,
                    train_loss=record.train_loss,
                    val_accuracy=record.val_accuracy,
                    val_f1=record.val_f1,
                    val_roc_auc=record.val_roc_auc,
                    val_log_loss=record.val_log_loss,
                    mean_update_norm=record.mean_update_norm,
                    clipped_fraction=record.clipped_fraction,
                    epsilon=record.epsilon,
                    n_clients=record.clients,
                )
            )
            session.commit()
            if on_round is not None:
                on_round(record)

        result = train_federated(train_ds, val_ds, cfg, on_round=persist_round)

        run = session.get(Run, run_id) or run
        run.status = "finished"
        run.finished_at = datetime.now(timezone.utc)
        run.duration_s = result.duration_s
        run.best_round = result.best_round
        run.rounds_run = len(result.history)
        run.val_accuracy = result.val_metrics.get("accuracy")
        run.val_f1 = result.val_metrics.get("f1")
        run.val_roc_auc = result.val_metrics.get("roc_auc")
        run.clients_json = json.dumps(result.clients)
        run.model_json = json.dumps(result.model.to_dict())
        run.scaler_json = json.dumps(
            {
                "feature_names": list(train_ds.feature_names),
                "mean": [float(v) for v in train_ds.mean],
                "std": [float(v) for v in train_ds.std],
            }
        )
        if result.privacy.get("enabled"):
            run.epsilon = float(result.privacy.get("epsilon", 0.0))
        session.commit()
    except Exception as exc:  # noqa: BLE001 - any failure must land in the DB
        session.rollback()
        failed = session.get(Run, run_id)
        if failed is not None:
            failed.status = "failed"
            failed.error = f"{type(exc).__name__}: {exc}"[:900]
            failed.finished_at = datetime.now(timezone.utc)
            session.commit()
        traceback.print_exc()
    finally:
        session.close()


def run_config_dict(run: Run) -> Dict[str, Any]:
    try:
        return json.loads(run.config_json or "{}")
    except json.JSONDecodeError:
        return {}
