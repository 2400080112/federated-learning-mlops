"""Regression tests for bugs found by auditing the code, not by writing features.

Each test here failed before its fix. They stay because the same mistake is easy
to reintroduce: version arithmetic over a mutable set, filenames built from a
timestamp, and a stage name standing in for a specific version.
"""

from __future__ import annotations

from conftest import finished_run, scored_version


def test_register_after_deleting_an_older_version(client, small_run):
    """Version numbering must survive deletions.

    Counting rows (`len(existing) + 1`) produced a duplicate version -- delete v1
    while v2 exists and the next registration claimed "v2" again, hitting the
    unique constraint and returning 500.
    """
    _, v1 = scored_version(client, {**small_run, "name": "first"})
    _, v2 = scored_version(client, {**small_run, "name": "second"})
    assert (v1["version"], v2["version"]) == ("v1", "v2")

    assert client.delete(f"/api/models/{v1['id']}").status_code == 204

    run = finished_run(client, {**small_run, "name": "third"})
    client.post(f"/api/runs/{run['id']}/evaluate")
    response = client.post(f"/api/runs/{run['id']}/register", json={"stage": "staging"})
    assert response.status_code == 201, response.text
    assert response.json()["version"] == "v3"


def test_version_numbers_are_not_reused_after_deleting_the_newest(client, small_run):
    """A gap stays a gap: deleting v2 must not make the next version v2 again."""
    _, v1 = scored_version(client, {**small_run, "name": "first"})
    _, v2 = scored_version(client, {**small_run, "name": "second"})
    client.delete(f"/api/models/{v2['id']}")

    run = finished_run(client, {**small_run, "name": "third"})
    client.post(f"/api/runs/{run['id']}/evaluate")
    version = client.post(f"/api/runs/{run['id']}/register", json={}).json()
    assert version["version"] == "v3"
    assert v1["version"] == "v1"


def test_rapid_predictions_get_separate_csvs(client, small_run):
    """Two scoring jobs in the same second must not overwrite each other.

    The filename used a second-resolution timestamp, so the second job replaced
    the first job's CSV and both download links served the same data.
    """
    _, version = scored_version(client, small_run)
    client.post(f"/api/models/{version['id']}/promote", json={"stage": "production"})

    first = client.post("/api/predict", json={"limit": 5}).json()
    second = client.post("/api/predict", json={"limit": 9}).json()
    assert (first["n_rows"], second["n_rows"]) == (5, 9)

    first_csv = client.get(first["download_url"]).text
    second_csv = client.get(second["download_url"]).text
    assert len(first_csv.splitlines()) == 6   # 5 rows + header
    assert len(second_csv.splitlines()) == 10  # 9 rows + header
    assert first_csv != second_csv


def test_prediction_can_pin_an_exact_version(client, small_run):
    """Pinning a version must not silently resolve to the newest one in its stage.

    The predict form used to send the *stage* of a chosen version, so picking an
    older staging version scored with whichever version was newest.
    """
    _, v1 = scored_version(client, {**small_run, "name": "first"})
    _, v2 = scored_version(client, {**small_run, "name": "second"})
    # v2 is the newest staging version; v1 has been archived by that registration
    assert client.get(f"/api/models/{v2['id']}").json()["stage"] == "staging"

    pinned = client.post("/api/predict", json={"model_version_id": v1["id"], "limit": 5})
    assert pinned.status_code == 200
    assert pinned.json()["model_label"] == f"wdbc-logreg:{v1['version']}"

    newest = client.post("/api/predict", json={"stage": "staging", "limit": 5})
    assert newest.json()["model_label"] == f"wdbc-logreg:{v2['version']}"
