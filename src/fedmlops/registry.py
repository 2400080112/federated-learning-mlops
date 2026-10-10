"""Simple filesystem model registry with immutable run directories."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .model import ModelWeights


def save_model(
    weights: ModelWeights,
    metrics: dict,
    config: dict,
    feature_names: list[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    registry_dir = project_root / "registry"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_dir = registry_dir / "versions" / run_id
    version_dir.mkdir(parents=True, exist_ok=False)
    np.savez(
        version_dir / "model.npz",
        coefficients=weights.coefficients,
        intercept=weights.intercept,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
    )
    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "config": config,
        "feature_names": feature_names,
        "positive_class": "benign",
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    registry_dir.mkdir(parents=True, exist_ok=True)
    latest_tmp = registry_dir / "latest.json.tmp"
    latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2), encoding="utf-8")
    latest_tmp.replace(registry_dir / "latest.json")
    return version_dir


def load_latest_model() -> tuple[ModelWeights, dict]:
    project_root = Path(__file__).resolve().parents[2]
    registry_dir = project_root / "registry"
    pointer = json.loads((registry_dir / "latest.json").read_text(encoding="utf-8"))
    version_dir = registry_dir / "versions" / pointer["run_id"]
    with np.load(version_dir / "model.npz") as saved:
        weights = ModelWeights(saved["coefficients"], float(saved["intercept"]))
        scaler = {"mean": saved["scaler_mean"], "scale": saved["scaler_scale"]}
    metadata = json.loads((version_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["scaler"] = {key: value.tolist() for key, value in scaler.items()}
    return weights, metadata
