"""Pydantic schemas -- the API contract, and the validation for it.

Field bounds are here rather than in the UI: the browser form is a convenience,
the server is what enforces them. Values that would silently produce a useless
run (0 clients, negative rounds) are rejected with a 422 before any work starts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Partition = Literal["iid", "dirichlet"]
Stage = Literal["staging", "production", "archived"]


# --------------------------------------------------------------- training
class RunCreate(BaseModel):
    """Everything the train form can set. All of it has a sane default."""

    name: str = Field(default="", max_length=120)
    n_clients: int = Field(default=8, ge=2, le=64)
    clients_per_round: Optional[int] = Field(default=None, ge=1, le=64)
    rounds: int = Field(default=25, ge=1, le=500)
    local_epochs: int = Field(default=3, ge=1, le=50)
    batch_size: int = Field(default=16, ge=1, le=512)
    learning_rate: float = Field(default=0.15, gt=0.0, le=5.0)
    l2: float = Field(default=0.001, ge=0.0, le=1.0)
    partition: Partition = "iid"
    dirichlet_alpha: float = Field(default=0.5, gt=0.0, le=10.0)
    seed: int = Field(default=7, ge=0, le=2**31 - 1)

    privacy_enabled: bool = False
    dp_epsilon: Optional[float] = Field(default=None, gt=0.0, le=1000.0)
    dp_sigma: Optional[float] = Field(default=None, gt=0.0, le=1000.0)
    dp_clip: float = Field(default=1.0, gt=0.0, le=100.0)
    dp_delta: float = Field(default=1e-5, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check(self) -> "RunCreate":
        if self.clients_per_round is not None and self.clients_per_round > self.n_clients:
            raise ValueError("clients_per_round cannot exceed n_clients")
        if self.privacy_enabled and self.dp_epsilon is None and self.dp_sigma is None:
            raise ValueError(
                "privacy is enabled but neither dp_epsilon nor dp_sigma was given"
            )
        if self.dp_epsilon is not None and self.dp_sigma is not None:
            raise ValueError("set dp_epsilon or dp_sigma, not both")
        return self

    @property
    def effective_clients_per_round(self) -> int:
        return self.clients_per_round or self.n_clients


class RoundPoint(BaseModel):
    round: int
    train_loss: float
    val_accuracy: float
    val_f1: float
    val_roc_auc: float
    val_log_loss: float
    mean_update_norm: float
    clipped_fraction: float
    epsilon: float
    n_clients: int


class RunSummary(BaseModel):
    """The list-view shape -- no per-round history, no model payload."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    status: str
    mode: str
    created_at: datetime
    finished_at: Optional[datetime] = None
    duration_s: Optional[float] = None
    best_round: Optional[int] = None
    rounds_run: Optional[int] = None
    n_clients: Optional[int] = None
    partition: Optional[str] = None
    privacy_enabled: bool = False
    epsilon: Optional[float] = None
    val_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_f1: Optional[float] = None
    test_roc_auc: Optional[float] = None
    is_scored: bool = False
    error: Optional[str] = None


class RunDetail(RunSummary):
    config: Dict[str, Any] = Field(default_factory=dict)
    clients: List[Dict[str, Any]] = Field(default_factory=list)
    history: List[RoundPoint] = Field(default_factory=list)
    test_metrics: Dict[str, float] = Field(default_factory=dict)
    noise_multiplier: Optional[float] = None
    clip_norm: Optional[float] = None
    registered_versions: List[str] = Field(default_factory=list)


class EvaluateResult(BaseModel):
    run_id: str
    test: Dict[str, float]


# --------------------------------------------------------------- registry
class RegisterRequest(BaseModel):
    name: str = Field(default="wdbc-logreg", max_length=64)
    stage: Stage = "staging"
    description: str = Field(default="", max_length=500)


class PromoteRequest(BaseModel):
    stage: Stage = "production"
    min_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ModelVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    version: str
    stage: str
    created_at: datetime
    updated_at: datetime
    run_id: Optional[str] = None
    description: str = ""
    test_accuracy: Optional[float] = None
    test_f1: Optional[float] = None
    test_roc_auc: Optional[float] = None
    epsilon: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------- predict
class PredictRequest(BaseModel):
    model_version_id: Optional[int] = None
    stage: Stage = "production"
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    use_bundled_dataset: bool = True
    limit: Optional[int] = Field(default=None, ge=1, le=569)


class PredictionOut(BaseModel):
    id: int
    created_at: datetime
    model_label: str
    source_label: str
    threshold: float
    n_rows: int
    summary: Dict[str, Any] = Field(default_factory=dict)
    preview: List[Dict[str, Any]] = Field(default_factory=list)
    download_url: str


# --------------------------------------------------------------- meta
class Stats(BaseModel):
    total_runs: int
    finished_runs: int
    running_runs: int
    failed_runs: int
    best_test_accuracy: Optional[float] = None
    best_run_id: Optional[str] = None
    production_model: Optional[ModelVersionOut] = None
    model_versions: int
    predictions: int
    dataset_rows: int
    dataset_features: int
    total_rounds: int


class HealthOut(BaseModel):
    status: str
    version: str
    database: str


class DatasetPreview(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]
    n_rows: int
    n_features: int
    class_balance: Dict[str, int]
