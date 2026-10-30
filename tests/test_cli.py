"""The terminal interface: same services, same rules, no HTTP.

Runs against a temporary SQLite file via FEDFL_DATABASE_URL, so it exercises the
real configuration path rather than a test double.
"""

from __future__ import annotations

import re

import pytest

from app.cli import main


@pytest.fixture
def cli_db(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDFL_DATABASE_URL", f"sqlite:///{tmp_path / 'cli.db'}")
    return tmp_path


def _run_id(output: str) -> str:
    match = re.search(r"run-\d{8}-\d{6}-[0-9a-f]{4}", output)
    assert match, f"no run id in output:\n{output}"
    return match.group(0)


def _version_id(output: str) -> int:
    match = re.search(r"--version-id (\d+)", output)
    assert match, f"no version id in output:\n{output}"
    return int(match.group(1))


def test_full_lifecycle_from_the_terminal(cli_db, capsys):
    assert main(["train", "--clients", "4", "--rounds", "3", "--name", "cli"]) == 0
    trained = capsys.readouterr().out
    run_id = _run_id(trained)
    assert "round   3" in trained          # progress printed per round
    assert "val " in trained

    assert main(["evaluate", "--run", run_id]) == 0
    assert "accuracy" in capsys.readouterr().out

    assert main(["register", "--run", run_id]) == 0
    registered = capsys.readouterr().out
    version_id = _version_id(registered)
    assert "wdbc-logreg:v1" in registered

    assert main(["promote", "--version-id", str(version_id), "--min-accuracy", "0.5"]) == 0
    assert "production" in capsys.readouterr().out

    assert main(["predict", "--limit", "10"]) == 0
    predicted = capsys.readouterr().out
    assert "scored 10 rows" in predicted

    assert main(["list-models"]) == 0
    assert "production" in capsys.readouterr().out


def test_promotion_gate_refuses_a_weak_model(cli_db, capsys):
    main(["train", "--clients", "4", "--rounds", "3"])
    run_id = _run_id(capsys.readouterr().out)
    main(["evaluate", "--run", run_id])
    capsys.readouterr()
    main(["register", "--run", run_id])
    version_id = _version_id(capsys.readouterr().out)

    assert main(["promote", "--version-id", str(version_id), "--min-accuracy", "0.999"]) == 2
    assert "promotion blocked" in capsys.readouterr().err


def test_unknown_run_is_reported_not_raised(cli_db, capsys):
    assert main(["evaluate", "--run", "run-does-not-exist"]) == 2
    assert "no such run" in capsys.readouterr().err


def test_predict_without_a_production_model(cli_db, capsys):
    assert main(["predict", "--limit", "5"]) == 2
    assert "production" in capsys.readouterr().err


def test_dp_run_from_the_terminal(cli_db, capsys):
    assert main(["train", "--clients", "4", "--rounds", "4", "--dp-epsilon", "6"]) == 0
    output = capsys.readouterr().out
    assert "eps=" in output
    assert "privacy" in output
