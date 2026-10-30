"""Meta endpoints: health, dashboard stats, dataset preview, form defaults."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fedmlops.data import resolve_path

from .. import __version__
from ..db import PROJECT_ROOT
from ..deps import get_db
from ..models import ModelVersion, Prediction, RoundMetric, Run
from ..schemas import DatasetPreview, HealthOut, RunCreate, Stats
from ..services import registry_service
from ..services.serialize import model_to_out

router = APIRouter(prefix="/api", tags=["meta"])

DATASET = PROJECT_ROOT / "data" / "breast_cancer.csv"


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(status="ok", version=__version__, database="sqlite")


@router.get("/stats", response_model=Stats)
def stats(db: Session = Depends(get_db)) -> Stats:
    total = db.scalar(select(func.count()).select_from(Run)) or 0
    by_status = dict(
        db.execute(select(Run.status, func.count()).group_by(Run.status)).all()
    )
    rounds = db.scalar(select(func.count()).select_from(RoundMetric)) or 0
    versions = db.scalar(select(func.count()).select_from(ModelVersion)) or 0
    preds = db.scalar(select(func.count()).select_from(Prediction)) or 0

    best = db.scalars(
        select(Run).where(Run.test_accuracy.is_not(None)).order_by(Run.test_accuracy.desc())
    ).first()

    prod = registry_service.production_model(db)
    n_rows = n_features = 0
    if DATASET.exists():
        df = pd.read_csv(DATASET, nrows=1)
        n_features = len([c for c in df.columns if c not in ("sample_id", "target", "target_name")])
        n_rows = int(sum(1 for _ in DATASET.open("r", encoding="utf-8")) - 1)

    return Stats(
        total_runs=total,
        finished_runs=by_status.get("finished", 0),
        running_runs=by_status.get("running", 0) + by_status.get("queued", 0),
        failed_runs=by_status.get("failed", 0),
        best_test_accuracy=best.test_accuracy if best else None,
        best_run_id=best.id if best else None,
        production_model=model_to_out(prod) if prod else None,
        model_versions=versions,
        predictions=preds,
        dataset_rows=n_rows,
        dataset_features=n_features,
        total_rounds=rounds,
    )


@router.get("/dataset/preview", response_model=DatasetPreview)
def dataset_preview(limit: int = Query(default=10, ge=1, le=100)) -> DatasetPreview:
    path = resolve_path(DATASET)
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("sample_id", "target", "target_name")]
    head = df.head(limit).to_dict(orient="records")
    for row in head:
        for key, value in list(row.items()):
            if isinstance(value, float):
                row[key] = round(value, 4)
    balance = {
        "benign": int((df["target"] == 1).sum()),
        "malignant": int((df["target"] == 0).sum()),
    }
    return DatasetPreview(
        columns=list(df.columns),
        rows=head,
        n_rows=int(len(df)),
        n_features=len(feature_cols),
        class_balance=balance,
    )


@router.get("/config/defaults", response_model=Dict[str, Any])
def config_defaults() -> Dict[str, Any]:
    """The values the train form starts from -- one source of truth with the API."""
    return json.loads(RunCreate().model_dump_json())
