"""Run endpoints: create, list, inspect, evaluate, register, delete."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..deps import get_db
from ..models import Run
from ..schemas import EvaluateResult, ModelVersionOut, RegisterRequest, RunCreate, RunDetail, RunSummary
from ..services import registry_service, training
from ..services.serialize import model_to_out, run_to_detail, run_to_summary

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.post("", response_model=RunSummary, status_code=status.HTTP_201_CREATED)
def create_run(
    payload: RunCreate,
    request: Request,
    db: Session = Depends(get_db),
) -> RunSummary:
    """Queue a federated run and start it in the background.

    Returns immediately with the queued run; poll ``GET /api/runs/{id}`` for
    progress (the ``history`` array grows one entry per round).
    """
    run = training.create_run(db, payload)
    training.start_training(request.app.state.executor, request.app.state.session_factory, run.id)
    return run_to_summary(run)


@router.get("", response_model=List[RunSummary])
def list_runs(
    db: Session = Depends(get_db),
    limit: int = Query(default=100, ge=1, le=500),
    run_status: Optional[str] = Query(default=None, alias="status"),
    scored_only: bool = False,
) -> List[RunSummary]:
    stmt = select(Run).order_by(Run.created_at.desc()).limit(limit)
    if run_status:
        stmt = stmt.where(Run.status == run_status)
    if scored_only:
        stmt = stmt.where(Run.test_accuracy.is_not(None))
    return [run_to_summary(r) for r in db.scalars(stmt).all()]


@router.get("/{run_id}", response_model=RunDetail)
def get_run(run_id: str, db: Session = Depends(get_db)) -> RunDetail:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return run_to_detail(run)


@router.post("/{run_id}/evaluate", response_model=EvaluateResult)
def evaluate_run(run_id: str, db: Session = Depends(get_db)) -> EvaluateResult:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    scores = registry_service.evaluate_run(db, run)
    return EvaluateResult(run_id=run_id, test=scores)


@router.post(
    "/{run_id}/register",
    response_model=ModelVersionOut,
    status_code=status.HTTP_201_CREATED,
)
def register_run(
    run_id: str, payload: RegisterRequest, db: Session = Depends(get_db)
) -> ModelVersionOut:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    mv = registry_service.register_run(
        db, run, name=payload.name, stage=payload.stage, description=payload.description
    )
    db.refresh(mv)
    return model_to_out(mv)


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_run(run_id: str, db: Session = Depends(get_db)) -> None:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    if run.model_versions:
        raise HTTPException(
            status_code=409,
            detail=(
                "this run is the source of a registered model version "
                f"({', '.join(f'{m.name}:{m.version}' for m in run.model_versions)}); "
                "delete or re-point those versions first"
            ),
        )
    db.delete(run)
    db.commit()
