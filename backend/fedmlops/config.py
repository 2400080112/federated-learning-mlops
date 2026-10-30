"""Configuration objects for a federated run.

Configs are plain YAML so a run is reproducible from one file. Every field has
a default, so ``fedmlops train`` works with no arguments at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DataConfig:
    csv_path: str = "data/breast_cancer.csv"
    target_column: str = "target"
    drop_columns: tuple = ("sample_id", "target_name")
    test_size: float = 0.20
    val_size: float = 0.20
    seed: int = 7


@dataclass
class FederatedConfig:
    n_clients: int = 8
    clients_per_round: int = 8          # how many clients report in each round
    rounds: int = 25
    local_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 0.15
    l2: float = 0.001
    partition: str = "iid"              # "iid" or "dirichlet"
    dirichlet_alpha: float = 0.5        # label skew; lower = more non-IID
    min_client_samples: int = 10


@dataclass
class PrivacyConfig:
    enabled: bool = False
    clip_norm: float = 1.0              # per-client update clipping bound C
    noise_multiplier: float = 0.0       # sigma, as a multiple of C
    delta: float = 1e-5                 # target delta for the budget report
    max_epsilon: float = 0.0            # 0 = do not enforce a budget


@dataclass
class TrackingConfig:
    output_dir: str = "artifacts"
    experiment: str = "federated-demo"


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    seed: int = 7

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RunConfig":
        raw = dict(raw or {})

        def merged(defaults: Any, override: Dict[str, Any]) -> Dict[str, Any]:
            # Coerce to the declared field types. Needed because YAML happily
            # reads `1e-5` as a *string* (it wants `1.0e-5` for a float), and a
            # string delta would explode deep inside the privacy maths.
            base = asdict(defaults)
            for key, value in (override or {}).items():
                if key not in base:
                    continue
                current = base[key]
                if isinstance(current, bool):
                    base[key] = bool(value)
                elif isinstance(current, float):
                    base[key] = float(value)
                elif isinstance(current, int):
                    base[key] = int(value)
                elif isinstance(current, tuple):
                    base[key] = tuple(value)
                else:
                    base[key] = value
            return base

        return cls(
            data=DataConfig(**merged(DataConfig(), raw.get("data") or {})),
            federated=FederatedConfig(**merged(FederatedConfig(), raw.get("federated") or {})),
            privacy=PrivacyConfig(**merged(PrivacyConfig(), raw.get("privacy") or {})),
            tracking=TrackingConfig(**merged(TrackingConfig(), raw.get("tracking") or {})),
            seed=int(raw.get("seed", 7)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = cls.from_dict(raw)
        cfg.validate()
        return cfg

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    # ---- guardrails ----------------------------------------------------
    def validate(self) -> None:
        fed = self.federated
        if fed.partition not in {"iid", "dirichlet"}:
            raise ValueError(f"partition must be 'iid' or 'dirichlet', got {fed.partition!r}")
        if fed.n_clients < 2:
            raise ValueError("n_clients must be >= 2")
        if not 1 <= fed.clients_per_round <= fed.n_clients:
            raise ValueError("clients_per_round must be between 1 and n_clients")
        if fed.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.data.test_size <= 0 or self.data.val_size <= 0:
            raise ValueError("test_size and val_size must be > 0")
        if self.data.test_size + self.data.val_size >= 1:
            raise ValueError("test_size + val_size must be < 1")
        if self.privacy.enabled:
            if self.privacy.noise_multiplier <= 0:
                raise ValueError(
                    "privacy.enabled is true but noise_multiplier is 0 -- "
                    "that is not privacy-preserving; set a sigma > 0"
                )
            if self.privacy.clip_norm <= 0:
                raise ValueError("privacy.clip_norm must be > 0")
