"""Run endpoints: creation, validation, lifecycle, deletion."""

from __future__ import annotations

import pytest

from conftest import finished_run, wait_for_run


def test_health_and_stats_start_empty(client):
    assert client.get("/api/health").json()["status"] == "ok"
    stats = client.get("/api/stats").json()
    assert stats["total_runs"] == 0
    assert stats["production_model"] is None
    assert stats["dataset_rows"] == 569
    assert stats["dataset_features"] == 30


def test_create_run_runs_to_completion(client, small_run):
    created = client.post("/api/runs", json=small_run)
    assert created.status_code == 201
    run_id = created.json()["id"]

    detail = wait_for_run(client, run_id)
    assert detail["status"] == "finished", detail.get("error")
    assert len(detail["history"]) == 4
    assert 0.5 < detail["val_accuracy"] <= 1.0
    assert detail["test_metrics"] == {}          # not scored yet
    assert detail["is_scored"] is False
    # one row per round, in order, each carrying the metrics we chart
    assert [h["round"] for h in detail["history"]] == [1, 2, 3, 4]
    assert all("val_roc_auc" in h for h in detail["history"])


def test_run_persists_its_configuration(client, small_run):
    run = finished_run(client, small_run)
    config = client.get(f"/api/runs/{run['id']}").json()["config"]
    assert config["federated"]["n_clients"] == 4
    assert config["federated"]["rounds"] == 4
    assert config["privacy"]["enabled"] is False


def test_list_runs_and_filters(client, small_run):
    finished_run(client, small_run)
    other = finished_run(client, {**small_run, "name": "second"})
    assert len(client.get("/api/runs").json()) == 2
    assert len(client.get("/api/runs?status=finished").json()) == 2
    assert client.get("/api/runs?status=running").json() == []
    assert client.get("/api/runs?scored_only=true").json() == []
    client.post(f"/api/runs/{other['id']}/evaluate")
    assert len(client.get("/api/runs?scored_only=true").json()) == 1


def test_validation_rejects_impossible_configs(client):
    bad = [
        {"n_clients": 1},                              # fewer than 2 clients
        {"n_clients": 4, "rounds": 0},                 # no rounds
        {"n_clients": 4, "clients_per_round": 9},      # more clients than exist
        {"n_clients": 4, "learning_rate": 0},          # no learning
        {"n_clients": 4, "partition": "banana"},       # unknown partition
    ]
    for payload in bad:
        assert client.post("/api/runs", json=payload).status_code == 422, payload


def test_dp_requires_a_noise_knob(client):
    response = client.post("/api/runs", json={"n_clients": 4, "rounds": 2, "privacy_enabled": True})
    assert response.status_code == 422
    assert "dp_epsilon" in response.text or "dp_sigma" in response.text


def test_dp_run_reports_epsilon(client):
    run = finished_run(
        client,
        {"n_clients": 4, "rounds": 6, "privacy_enabled": True, "dp_epsilon": 6.0, "seed": 2},
    )
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["privacy_enabled"] is True
    assert detail["epsilon"] == pytest.approx(6.0, rel=0.05)
    assert detail["noise_multiplier"] > 0
    # the per-round epsilon grows as the budget is spent
    epsilons = [h["epsilon"] for h in detail["history"]]
    assert epsilons == sorted(epsilons)


def test_evaluate_then_register_then_promote(client, small_run):
    run = finished_run(client, small_run)

    scored = client.post(f"/api/runs/{run['id']}/evaluate")
    assert scored.status_code == 200
    assert scored.json()["test"]["accuracy"] > 0.5

    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["is_scored"] is True
    assert set(detail["test_metrics"]) >= {"accuracy", "f1", "roc_auc"}

    version = client.post(f"/api/runs/{run['id']}/register", json={"stage": "staging"})
    assert version.status_code == 201
    assert version.json()["version"] == "v1"
    assert version.json()["stage"] == "staging"

    promoted = client.post(
        f"/api/models/{version.json()['id']}/promote",
        json={"stage": "production", "min_accuracy": 0.5},
    )
    assert promoted.status_code == 200
    assert promoted.json()["stage"] == "production"
    assert client.get("/api/models/production").json()["version"] == "v1"


def test_delete_run_removes_it(client, small_run):
    run = finished_run(client, small_run)
    assert client.delete(f"/api/runs/{run['id']}").status_code == 204
    assert client.get(f"/api/runs/{run['id']}").status_code == 404
    assert client.get("/api/runs").json() == []


def test_delete_run_refuses_when_a_model_points_at_it(client, small_run):
    run = finished_run(client, small_run)
    client.post(f"/api/runs/{run['id']}/evaluate")
    client.post(f"/api/runs/{run['id']}/register", json={})
    response = client.delete(f"/api/runs/{run['id']}")
    assert response.status_code == 409
    assert "registered model version" in response.json()["detail"]


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/nope").status_code == 404
    assert client.post("/api/runs/nope/evaluate").status_code == 404
