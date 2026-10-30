"""Prediction endpoints: bundled data, uploads, download, and their failures."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from conftest import scored_version

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "breast_cancer.csv"


@pytest.fixture
def production(client, small_run):
    _, version = scored_version(client, small_run)
    client.post(f"/api/models/{version['id']}/promote", json={"stage": "production"})
    return version


def test_predict_with_the_bundled_dataset(client, production):
    response = client.post("/api/predict", json={"limit": 40, "threshold": 0.5})
    assert response.status_code == 200
    body = response.json()
    assert body["n_rows"] == 40
    assert body["model_label"] == "wdbc-logreg:v1"
    assert body["summary"]["predicted_benign"] + body["summary"]["predicted_malignant"] == 40
    # the bundled file has a target column, so accuracy and a confusion matrix appear
    assert "accuracy" in body["summary"]
    assert set(body["summary"]["confusion"]) == {"tp", "tn", "fp", "fn"}
    assert len(body["preview"]) <= 25


def test_prediction_is_downloadable_csv(client, production):
    body = client.post("/api/predict", json={"limit": 10}).json()
    response = client.get(body["download_url"])
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    frame = pd.read_csv(io.BytesIO(response.content))
    assert len(frame) == 10
    assert {"p_benign", "prediction", "prediction_label"} <= set(frame.columns)


def test_threshold_moves_predictions(client, production):
    strict = client.post("/api/predict", json={"limit": 100, "threshold": 0.9}).json()
    loose = client.post("/api/predict", json={"limit": 100, "threshold": 0.1}).json()
    assert strict["summary"]["predicted_benign"] <= loose["summary"]["predicted_benign"]


def test_upload_scores_a_csv(client, production):
    raw = DATASET.read_bytes()
    response = client.post(
        "/api/predict/upload",
        files={"file": ("rows.csv", raw, "text/csv")},
        data={"threshold": "0.5", "stage": "production"},
    )
    assert response.status_code == 200
    assert response.json()["n_rows"] == 569
    assert response.json()["source_label"] == "uploaded CSV"


def test_upload_without_feature_columns_is_rejected(client, production):
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    buffer = io.BytesIO()
    frame.to_csv(buffer, index=False)
    response = client.post(
        "/api/predict/upload",
        files={"file": ("bad.csv", buffer.getvalue(), "text/csv")},
        data={"threshold": "0.5"},
    )
    assert response.status_code == 409
    assert "missing" in response.json()["detail"]


def test_upload_rejects_a_bad_threshold(client, production):
    raw = DATASET.read_bytes()
    response = client.post(
        "/api/predict/upload",
        files={"file": ("rows.csv", raw, "text/csv")},
        data={"threshold": "1.5"},
    )
    assert response.status_code == 422


def test_predict_without_a_production_model_is_404(client):
    response = client.post("/api/predict", json={"limit": 5})
    assert response.status_code == 404
    assert "production" in response.json()["detail"]


def test_prediction_jobs_are_listed(client, production):
    client.post("/api/predict", json={"limit": 5})
    client.post("/api/predict", json={"limit": 6})
    jobs = client.get("/api/predictions").json()
    assert len(jobs) == 2
    assert all(job["download_url"].startswith("/api/predictions/") for job in jobs)


def test_dataset_preview(client):
    preview = client.get("/api/dataset/preview?limit=3").json()
    assert preview["n_rows"] == 569
    assert preview["n_features"] == 30
    assert preview["class_balance"] == {"benign": 357, "malignant": 212}
    assert len(preview["rows"]) == 3
