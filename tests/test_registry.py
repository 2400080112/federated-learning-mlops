
import json
from pathlib import Path

import numpy as np

import src.fedmlops.registry as registry
from src.fedmlops.model import ModelWeights


def test_save_and_load_model(monkeypatch, tmp_path):
    # Use a temporary registry, not the real project registry.
    monkeypatch.setattr(
        registry,
        "__file__",
        str(tmp_path / "src" / "fedmlops" / "registry.py"),
    )

    project_root = Path(registry.__file__).resolve().parents[2]
    registry_dir = project_root / "registry"
    registry_dir.mkdir(parents=True)

    weights = ModelWeights(
        coefficients=np.array([0.2, -0.4]),
        intercept=0.1,
    )
    metrics = {"accuracy": 0.95}
    config = {"learning_rate": 0.1}
    feature_names = ["feature_a", "feature_b"]
    scaler_mean = np.array([1.0, 2.0])
    scaler_scale = np.array([0.5, 1.5])

    version_dir = registry.save_model(
        weights,
        metrics,
        config,
        feature_names,
        scaler_mean,
        scaler_scale,
    )

    assert (version_dir / "model.npz").exists()
    assert (version_dir / "metadata.json").exists()
    assert (registry_dir / "latest.json").exists()

    saved_pointer = json.loads(
        (registry_dir / "latest.json").read_text(encoding="utf-8")
    )
    assert saved_pointer["run_id"] == version_dir.name

    loaded_weights, metadata = registry.load_latest_model()

    assert np.allclose(
        loaded_weights.coefficients, weights.coefficients
    )
    assert np.isclose(loaded_weights.intercept, weights.intercept)
    assert metadata["metrics"] == metrics
    assert metadata["config"] == config
    assert metadata["feature_names"] == feature_names
    assert np.allclose(
        metadata["scaler"]["mean"], scaler_mean
    )
    assert np.allclose(
        metadata["scaler"]["scale"], scaler_scale
    )
