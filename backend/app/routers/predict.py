"""Prediction endpoints: score a CSV with a registered model, list past runs of it."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..deps import get_db
from ..models import ModelVersion, Prediction
from ..schemas import PredictRequest, PredictionOut
from ..services import prediction_service as ps
from ..services import registry_service

router = APIRouter(prefix="/api", tags=["predict"])


def _resolve_model(db: Session, version_id: Optional[int], stage: str) -> ModelVersion:
    if version_id is not None:
        mv = db.get(ModelVersion, version_id)
        if mv is None:
            raise HTTPException(status_code=404, detail=f"model version {version_id} not found")
        return mv
    mv = db.scalars(
        select(ModelVersion)
        .where(ModelVersion.stage == stage)
        .order_by(ModelVersion.id.desc())
    ).first()
    if mv is None:
        raise HTTPException(
            status_code=404,
            detail=f"no model version in stage {stage!r}. Register and promote one first.",
        )
    return mv


@router.post("/predict", response_model=PredictionOut)
def predict_with_bundled(
    payload: PredictRequest, db: Session = Depends(get_db)
) -> PredictionOut:
    """Score the bundled dataset (or the first N rows of it) with a model version."""
    mv = _resolve_model(db, payload.model_version_id, payload.stage)
    record = ps.predict(
        db,
        mv,
        use_bundled=payload.use_bundled_dataset,
        threshold=payload.threshold,
        limit=payload.limit,
    )
    return PredictionOut(**ps.to_out(record))


@router.post("/predict/upload", response_model=PredictionOut)
async def predict_with_upload(
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    stage: str = Form("production"),
    model_version_id: Optional[int] = Form(None),
) -> PredictionOut:
    """Score an uploaded CSV. The header must contain the 30 feature columns."""
    if not (0.0 < threshold < 1.0):
        raise HTTPException(status_code=422, detail="threshold must be between 0 and 1")
    mv = _resolve_model(db, model_version_id, stage)
    raw = await file.read()
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file larger than 5 MB")
    record = ps.predict(db, mv, csv_bytes=raw, threshold=threshold)
    return PredictionOut(**ps.to_out(record))


@router.get("/predictions", response_model=List[PredictionOut])
def list_predictions(db: Session = Depends(get_db), limit: int = 25) -> List[PredictionOut]:
    rows = db.scalars(
        select(Prediction).order_by(Prediction.created_at.desc()).limit(max(1, min(limit, 100)))
    ).all()
    return [PredictionOut(**ps.to_out(r)) for r in rows]


@router.get("/predictions/{prediction_id}", response_model=PredictionOut)
def get_prediction(prediction_id: int, db: Session = Depends(get_db)) -> PredictionOut:
    record = db.get(Prediction, prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"prediction {prediction_id} not found")
    return PredictionOut(**ps.to_out(record))


@router.get("/predictions/{prediction_id}/download")
def download_prediction(prediction_id: int, db: Session = Depends(get_db)) -> FileResponse:
    record = db.get(Prediction, prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"prediction {prediction_id} not found")
    try:
        path = ps.output_file(record)
    except ps.ServiceError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    return FileResponse(
        path, media_type="text/csv", filename=f"predictions-{prediction_id}.csv"
    )
