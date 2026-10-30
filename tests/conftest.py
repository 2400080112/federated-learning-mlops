"""Fixtures: an isolated in-memory database and app per test."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# app/main.py builds a module-level `app` at import time, which would create (and
# touch) the project's real data/app.db during a test run. Point the import-time
# engine at a throwaway file first; individual tests still get their own
# in-memory database via create_app("sqlite://").
_IMPORT_DB = Path(tempfile.mkdtemp(prefix="fedfl-test-")) / "import-time.db"
os.environ.setdefault("FEDFL_DATABASE_URL", f"sqlite:///{_IMPORT_DB}")

from app.main import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path) -> TestClient:
    """Fresh app + its own SQLite file per test, so tests cannot leak state.

    A file rather than ``:memory:`` on purpose: an in-memory database has to share
    one connection (StaticPool), and the background training thread and the
    request thread using the same connection at once is a data race that shows up
    as intermittent "database is locked" failures. A file gets a connection per
    session, which is what production does too.
    """
    app = create_app(f"sqlite:///{tmp_path / 'test.db'}")
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
