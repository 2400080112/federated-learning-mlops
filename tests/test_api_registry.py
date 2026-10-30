"""Registry rules: the measurement gate, promotion, archiving, rollback."""

from __future__ import annotations

from conftest import finished_run, scored_version


def test_register_requires_a_test_score(client, small_run):
    run = finished_run(client, small_run)
    response = client.post(f"/api/runs/{run['id']}/register", json={})
    assert response.status_code == 409
    assert "evaluate" in response.json()["detail"].lower()


def test_register_requires_a_trained_model(client):
    run = client.post("/api/runs", json={"n_clients": 4, "rounds": 1}).json()
    response = client.post(f"/api/runs/{run['id']}/register", json={})
    assert response.status_code == 409


def test_versions_increment_and_carry_their_source(client, small_run):
    run_a, v1 = scored_version(client, {**small_run, "name": "a"})
    run_b, v2 = scored_version(client, {**small_run, "name": "b"})
    assert (v1["version"], v2["version"]) == ("v1", "v2")
    assert v1["run_id"] == run_a["id"] and v2["run_id"] == run_b["id"]
    assert v1["metadata"]["source_run"] == run_a["id"]
    assert client.get("/api/models").json()[0]["version"] == "v2"  # newest first


def test_promotion_gate_blocks_a_weak_version(client, small_run):
    _, version = scored_version(client, small_run)
    response = client.post(
        f"/api/models/{version['id']}/promote",
        json={"stage": "production", "min_accuracy": 0.999},
    )
    assert response.status_code == 409
    assert "promotion blocked" in response.json()["detail"]
    assert client.get("/api/models/production").status_code == 404


def test_promoting_archives_the_previous_holder(client, small_run):
    _, v1 = scored_version(client, {**small_run, "name": "first"})
    _, v2 = scored_version(client, {**small_run, "name": "second"})

    client.post(f"/api/models/{v1['id']}/promote", json={"stage": "production"})
    assert client.get("/api/models/production").json()["version"] == "v1"

    client.post(f"/api/models/{v2['id']}/promote", json={"stage": "production"})
    versions = {v["version"]: v["stage"] for v in client.get("/api/models").json()}
    assert versions == {"v1": "archived", "v2": "production"}


def test_rollback_is_promoting_the_older_version_again(client, small_run):
    _, v1 = scored_version(client, {**small_run, "name": "first"})
    _, v2 = scored_version(client, {**small_run, "name": "second"})
    client.post(f"/api/models/{v1['id']}/promote", json={"stage": "production"})
    client.post(f"/api/models/{v2['id']}/promote", json={"stage": "production"})
    assert client.get("/api/models/production").json()["version"] == "v2"

    client.post(f"/api/models/{v1['id']}/promote", json={"stage": "production"})
    assert client.get("/api/models/production").json()["version"] == "v1"
    # the artifact itself never changed -- only the stage pointer
    assert client.get(f"/api/models/{v1['id']}").json()["stage"] == "production"


def test_staging_can_hold_one_version_per_name(client, small_run):
    _, v1 = scored_version(client, {**small_run, "name": "first"})
    _, v2 = scored_version(client, {**small_run, "name": "second"})
    # both were registered as staging; the second registration demoted the first
    stages = {v["version"]: v["stage"] for v in client.get("/api/models").json()}
    assert stages["v2"] == "staging"
    assert stages["v1"] == "archived"


def test_cannot_delete_the_production_model(client, small_run):
    _, version = scored_version(client, small_run)
    client.post(f"/api/models/{version['id']}/promote", json={"stage": "production"})
    response = client.delete(f"/api/models/{version['id']}")
    assert response.status_code == 409
    assert "production" in response.json()["detail"]


def test_can_delete_an_archived_version(client, small_run):
    _, v1 = scored_version(client, {**small_run, "name": "first"})
    _, _ = scored_version(client, {**small_run, "name": "second"})
    assert client.delete(f"/api/models/{v1['id']}").status_code == 204
    assert [v["version"] for v in client.get("/api/models").json()] == ["v2"]


def test_stats_reflect_the_pipeline(client, small_run):
    run, version = scored_version(client, small_run)
    client.post(f"/api/models/{version['id']}/promote", json={"stage": "production"})
    stats = client.get("/api/stats").json()
    assert stats["total_runs"] == 1
    assert stats["finished_runs"] == 1
    assert stats["model_versions"] == 1
    assert stats["total_rounds"] == 4
    assert stats["production_model"]["version"] == "v1"
    assert stats["best_run_id"] == run["id"]
    assert stats["best_test_accuracy"] > 0.5
