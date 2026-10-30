"""Registry endpoints: list versions, promote, roll back, read production."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..deps import get_db
from ..models import ModelVersion
from ..schemas import ModelVersionOut, PromoteRequest
from ..services import registry_service
from ..services.serialize import model_to_out

router = APIRouter(prefix="/api/models", tags=["registry"])


@router.get("", response_model=List[ModelVersionOut])
def list_versions(
    db: Session = Depends(get_db),
    name: str = Query(default="wdbc-logreg"),
    stage: str | None = Query(default=None),
) -> List[ModelVersionOut]:
    stmt = select(ModelVersion).where(ModelVersion.name == name).order_by(ModelVersion.id.desc())
    if stage:
        stmt = stmt.where(ModelVersion.stage == stage)
    return [model_to_out(mv) for mv in db.scalars(stmt).all()]


@router.get("/production", response_model=ModelVersionOut)
def get_production(
    db: Session = Depends(get_db), name: str = Query(default="wdbc-logreg")
) -> ModelVersionOut:
    mv = registry_service.production_model(db, name)
    if mv is None:
        raise HTTPException(
            status_code=404,
            detail=f"no version of {name!r} is in production. Register and promote one first.",
        )
    return model_to_out(mv)


@router.get("/{version_id}", response_model=ModelVersionOut)
def get_version(version_id: int, db: Session = Depends(get_db)) -> ModelVersionOut:
    mv = db.get(ModelVersion, version_id)
    if mv is None:
        raise HTTPException(status_code=404, detail=f"model version {version_id} not found")
    return model_to_out(mv)


@router.post("/{version_id}/promote", response_model=ModelVersionOut)
def promote_version(
    version_id: int, payload: PromoteRequest, db: Session = Depends(get_db)
) -> ModelVersionOut:
    """Move a version into a stage. ``min_accuracy`` is the promotion gate."""
    mv = db.get(ModelVersion, version_id)
    if mv is None:
        raise HTTPException(status_code=404, detail=f"model version {version_id} not found")
    updated = registry_service.promote(
        db, mv, stage=payload.stage, min_accuracy=payload.min_accuracy
    )
    db.refresh(updated)
    return model_to_out(updated)


@router.delete("/{version_id}", status_code=204)
def delete_version(version_id: int, db: Session = Depends(get_db)) -> None:
    mv = db.get(ModelVersion, version_id)
    if mv is None:
        raise HTTPException(status_code=404, detail=f"model version {version_id} not found")
    if mv.stage == "production":
        raise HTTPException(
            status_code=409,
            detail="refusing to delete the production model; promote another version first",
        )
    db.delete(mv)
    db.commit()
