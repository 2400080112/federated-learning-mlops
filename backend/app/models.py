"""ORM tables: runs, per-round metrics, registry versions, predictions.

Three ideas worth noticing, because they are what makes this a *database* rather
than a JSON folder:

* ``round_metrics`` is one row per round per run -- the training curve is
  queryable, so the UI can chart any run without re-reading a file.
* ``model_versions`` has a uniqueness constraint on ``(name, version)`` and
  stores its stage, so promotion is an UPDATE, and a rollback is another UPDATE.
* ``runs.model_json`` holds the trained parameters. Small enough to live in a
  column here; in a real system this would be an object store with a URI.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    mode: Mapped[str] = mapped_column(String(24), default="federated")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # configuration as submitted, so a run can be reproduced or replayed
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    clients_json: Mapped[str] = mapped_column(Text, default="[]")
    scaler_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    best_round: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rounds_run: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    val_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    val_f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    val_roc_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    test_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    test_f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    test_roc_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    test_log_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    test_precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    test_recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    privacy_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    epsilon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    noise_multiplier: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clip_norm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    metrics: Mapped[List["RoundMetric"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="RoundMetric.round_no",
        lazy="selectin",
    )
    model_versions: Mapped[List["ModelVersion"]] = relationship(back_populates="run")

    @property
    def is_scored(self) -> bool:
        return self.test_accuracy is not None


class RoundMetric(Base):
    __tablename__ = "round_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    round_no: Mapped[int] = mapped_column(Integer)

    train_loss: Mapped[float] = mapped_column(Float, default=0.0)
    val_accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    val_f1: Mapped[float] = mapped_column(Float, default=0.0)
    val_roc_auc: Mapped[float] = mapped_column(Float, default=0.0)
    val_log_loss: Mapped[float] = mapped_column(Float, default=0.0)
    mean_update_norm: Mapped[float] = mapped_column(Float, default=0.0)
    clipped_fraction: Mapped[float] = mapped_column(Float, default=0.0)
    epsilon: Mapped[float] = mapped_column(Float, default=0.0)
    n_clients: Mapped[int] = mapped_column(Integer, default=0)

    run: Mapped[Run] = relationship(back_populates="metrics")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_name_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), default="wdbc-logreg", index=True)
    version: Mapped[str] = mapped_column(String(16))
    stage: Mapped[str] = mapped_column(String(24), default="staging", index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
    )
    description: Mapped[str] = mapped_column(Text, default="")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    model_json: Mapped[str] = mapped_column(Text, default="{}")
    scaler_json: Mapped[str] = mapped_column(Text, default="{}")

    test_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    test_f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    test_roc_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    epsilon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    run: Mapped[Optional[Run]] = relationship(back_populates="model_versions")


class ModelCounter(Base):
    """Monotonic version counter per model name.

    A version number is an identity, so it must never be handed out twice -- not
    even after the version holding it is deleted. Deriving the next number from
    the rows that currently exist cannot guarantee that; a counter can.
    """

    __tablename__ = "model_counters"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    last_version: Mapped[int] = mapped_column(Integer, default=0)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    model_version_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True
    )
    model_label: Mapped[str] = mapped_column(String(64), default="")
    source_label: Mapped[str] = mapped_column(String(128), default="")
    threshold: Mapped[float] = mapped_column(Float, default=0.5)
    n_rows: Mapped[int] = mapped_column(Integer, default=0)
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    preview_json: Mapped[str] = mapped_column(Text, default="[]")
    output_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
