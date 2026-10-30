"""Populate a fresh database with a few representative runs.

    cd backend && python -m app.seed           # add the demo runs
    cd backend && python -m app.seed --reset   # wipe data/app.db first

Why a script instead of clicking: the dashboard is more useful when it has a
history to show, and this doubles as a smoke test of the whole pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .db import DEFAULT_DB_PATH, init_db, make_engine, make_session_factory
from .models import Run
from .schemas import RunCreate
from .services import prediction_service, registry_service, training

EXPERIMENTS = [
    {
        "name": "8 clients · iid · 25 rounds",
        "n_clients": 8, "rounds": 25, "seed": 7,
    },
    {
        "name": "non-IID · dirichlet α=0.3",
        "n_clients": 10, "rounds": 30, "partition": "dirichlet",
        "dirichlet_alpha": 0.3, "seed": 11,
    },
    {
        "name": "DP · ε=8 · 20 rounds",
        "n_clients": 10, "rounds": 20, "privacy_enabled": True,
        "dp_epsilon": 8.0, "seed": 5,
    },
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the demo database")
    parser.add_argument("--reset", action="store_true", help="delete the SQLite file first")
    args = parser.parse_args(argv)

    if args.reset and DEFAULT_DB_PATH.exists():
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(DEFAULT_DB_PATH) + suffix)
            if path.exists():
                path.unlink()
        print(f"removed {DEFAULT_DB_PATH}")

    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    session = session_factory()
    try:
        created_ids: list[str] = []
        for payload in EXPERIMENTS:
            run = training.create_run(session, RunCreate(**payload))
            print(f"training {run.name} …")
            training.run_training(session_factory, run.id)
            session.expire_all()
            run = session.get(Run, run.id)
            if run.status != "finished":
                print(f"  failed: {run.error}", file=sys.stderr)
                continue
            registry_service.evaluate_run(session, run)
            print(f"  {run.id} · val {run.val_accuracy:.4f} · test {run.test_accuracy:.4f}")
            created_ids.append(run.id)
    finally:
        session.close()

    # register everything, promote the best non-DP run to production
    session = session_factory()
    try:
        versions = []
        for run_id in created_ids:
            run = session.get(Run, run_id)
            if run is None:
                continue
            mv = registry_service.register_run(
                session, run, stage="staging",
                description=f"seeded from {run.name}",
            )
            versions.append(mv)
            print(f"registered {mv.name}:{mv.version} (test {mv.test_accuracy:.4f})")

        plain = [v for v in versions if not v.epsilon]
        best = max(plain or versions, key=lambda v: v.test_accuracy or 0)
        registry_service.promote(session, best, stage="production")
        print(f"promoted {best.name}:{best.version} to production")

        record = prediction_service.predict(session, best, use_bundled=True, threshold=0.5)
        print(f"scored {record.n_rows} rows -> {record.output_path}")
    finally:
        session.close()

    print("\ndone. start the server and open http://127.0.0.1:8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
