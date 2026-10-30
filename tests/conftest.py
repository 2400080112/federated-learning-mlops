"""Fixtures: an isolated in-memory database and app per test."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.main import create_app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    """Fresh app + in-memory SQLite per test, so tests cannot leak state."""
    app = create_app("sqlite://")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def small_run():
    """The smallest run worth waiting for: 4 clients, 4 rounds."""
    return {
        "name": "test run",
        "n_clients": 4,
        "rounds": 4,
        "local_epochs": 2,
        "batch_size": 16,
        "learning_rate": 0.2,
        "seed": 3,
    }


def wait_for_run(client: TestClient, run_id: str, timeout: float = 60.0) -> dict:
    """Poll until the background training job leaves queued/running."""
    deadline = time.time() + timeout
    detail = {}
    while time.time() < deadline:
        detail = client.get(f"/api/runs/{run_id}").json()
        if detail["status"] in ("finished", "failed"):
            return detail
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish in {timeout}s: {detail.get('status')}")


def finished_run(client: TestClient, payload: dict) -> dict:
    run_id = client.post("/api/runs", json=payload).json()["id"]
    return wait_for_run(client, run_id)


def scored_version(client: TestClient, payload: dict) -> tuple[dict, dict]:
    """Train -> evaluate -> register. Returns (run, model_version)."""
    run = finished_run(client, payload)
    client.post(f"/api/runs/{run['id']}/evaluate")
    version = client.post(
        f"/api/runs/{run['id']}/register", json={"stage": "staging"}
    ).json()
    return run, version
