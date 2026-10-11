
import json
from pathlib import Path

import src.fedmlops.pipeline as pipeline


def test_pipeline_completes_and_saves_outputs(monkeypatch, tmp_path):
    # Redirect generated artifacts and model registry to a temporary folder.
    monkeypatch.setattr(
        pipeline,
        "__file__",
        str(tmp_path / "src" / "fedmlops" / "pipeline.py"),
    )

    def fake_save_model(
        weights, metrics, config, feature_names, scaler_mean, scaler_scale
    ):
        version_dir = tmp_path / "test_registry" / "versions" / "test-version"
        version_dir.mkdir(parents=True)
        (version_dir / "metadata.json").write_text(
            json.dumps({"metrics": metrics}),
            encoding="utf-8",
        )
        return version_dir

    monkeypatch.setattr(pipeline, "save_model", fake_save_model)

    config = {
        "num_clients": 2,
        "test_size": 0.2,
        "random_seed": 42,
        "rounds": 2,
        "local_epochs": 1,
        "learning_rate": 0.1,
        "l2_strength": 0.001,
    }

    version_dir = pipeline.run_pipeline(config)

    assert version_dir.exists()
    assert (version_dir / "metadata.json").exists()

    project_root = Path(pipeline.__file__).resolve().parents[2]
    run_dirs = list((project_root / "artifacts" / "runs").iterdir())

    assert len(run_dirs) == 1

    saved_config = json.loads(
        (run_dirs[0] / "config.json").read_text(encoding="utf-8")
    )
    assert saved_config["rounds"] == 2

    metrics_lines = (
        run_dirs[0] / "metrics.csv"
    ).read_text(encoding="utf-8").splitlines()

    assert len(metrics_lines) == 3  # Header plus two training rounds
