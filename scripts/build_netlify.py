"""Build the static Netlify site from a real, seeded run.

Netlify cannot host this app as it runs locally -- there is no Python runtime for
a long-lived FastAPI process, no writable disk for SQLite, and nowhere for a
background training thread to live. So the static build does something honest
instead: it trains for real, dumps the API's own JSON responses into a snapshot,
and ships a frontend that replays the read paths from it. Predictions are not
replayed -- the model is 31 numbers, so the forward pass runs in the browser
(frontend/js/demo.js) and the threshold slider genuinely changes the output.

    python scripts/build_netlify.py                 # -> netlify/site/
    python scripts/build_netlify.py --api-base https://my-api.example.com

With ``--api-base`` the site points at a live API instead of the snapshot, which
is the other way to deploy: static frontend on Netlify, FastAPI elsewhere.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.main import create_app  # noqa: E402
from app.models import ModelVersion, Run  # noqa: E402
from app.schemas import RunCreate  # noqa: E402
from app.services import registry_service, training  # noqa: E402

FRONTEND = ROOT / "frontend"
OUT = ROOT / "netlify" / "site"
DATASET = ROOT / "data" / "breast_cancer.csv"

EXPERIMENTS = [
    {"name": "8 clients · iid · 25 rounds", "n_clients": 8, "rounds": 25, "seed": 7},
    {"name": "non-IID · dirichlet α=0.3", "n_clients": 10, "rounds": 30,
     "partition": "dirichlet", "dirichlet_alpha": 0.3, "seed": 11},
    {"name": "DP · ε=8 · 20 rounds", "n_clients": 10, "rounds": 20,
     "privacy_enabled": True, "dp_epsilon": 8.0, "seed": 5},
]


def seed(db_path: Path) -> None:
    """Train three runs, score them, register them, promote the best."""
    app = create_app(f"sqlite:///{db_path}")
    factory = app.state.session_factory
    session = factory()
    created: list[str] = []
    try:
        for payload in EXPERIMENTS:
            run = training.create_run(session, RunCreate(**payload))
            print(f"  training {run.name} …", flush=True)
            training.run_training(factory, run.id)
            session.expire_all()
            run = session.get(Run, run.id)
            if run.status != "finished":
                raise SystemExit(f"training failed: {run.error}")
            registry_service.evaluate_run(session, run)
            print(f"    val {run.val_accuracy:.4f} · test {run.test_accuracy:.4f}")
            created.append(run.id)
    finally:
        session.close()

    session = factory()
    try:
        versions = []
        for run_id in created:
            run = session.get(Run, run_id)
            versions.append(
                registry_service.register_run(
                    session, run, stage="staging", description=f"seeded from {run.name}"
                )
            )
        plain = [v for v in versions if not v.epsilon] or versions
        best = max(plain, key=lambda v: v.test_accuracy or 0)
        registry_service.promote(session, best, stage="production")
        print(f"  production: {best.name}:{best.version} (test {best.test_accuracy:.4f})")
    finally:
        session.close()


def snapshot(db_path: Path) -> dict:
    """Capture the API's own responses -- so the demo payload cannot drift."""
    app = create_app(f"sqlite:///{db_path}")
    client = TestClient(app)

    runs = client.get("/api/runs").json()
    run_details = [client.get(f"/api/runs/{r['id']}").json() for r in runs]
    models = client.get("/api/models").json()

    # one prediction job so the history table is not empty
    client.post("/api/predict", json={"stage": "production", "limit": 100})

    payload = {
        "version": client.get("/api/health").json()["version"],
        "generated_from": "scripts/build_netlify.py",
        "stats": client.get("/api/stats").json(),
        "runs": run_details,
        "models": models,
        "predictions": client.get("/api/predictions").json(),
        "dataset": client.get("/api/dataset/preview?limit=10").json(),
        "defaults": client.get("/api/config/defaults").json(),
    }

    # the model itself, for in-browser inference
    session = app.state.session_factory()
    try:
        production = session.scalars(
            select(ModelVersion).where(ModelVersion.stage == "production")
        ).first()
        if production is None:
            raise SystemExit("no production model to export")
        model = json.loads(production.model_json)
        scaler = json.loads(production.scaler_json)
    finally:
        session.close()

    frame = pd.read_csv(DATASET)
    features = scaler["feature_names"]
    rows = frame[features].to_numpy(dtype=float)
    payload["model_math"] = {
        "params": model["params"],
        "mean": scaler["mean"],
        "std": scaler["std"],
        "target_index": len(features),
        "sample_id_index": len(features) + 1,
    }
    # each row: 30 features, then target, then sample_id
    payload["dataset_rows"] = [
        [round(float(v), 6) for v in row] + [int(frame["target"].iloc[i]), int(frame["sample_id"].iloc[i])]
        for i, row in enumerate(rows)
    ]
    return payload


def build(api_base: str = "") -> None:
    db_path = Path(tempfile.mkdtemp(prefix="fedfl-netlify-")) / "demo.db"
    print("seeding a real run for the snapshot …")
    seed(db_path)

    print("capturing API responses …")
    payload = snapshot(db_path)

    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(FRONTEND, OUT)

    config = {"demo": not api_base, "apiBase": api_base}
    index = (OUT / "index.html").read_text(encoding="utf-8")
    index = index.replace(
        '  <script type="module" src="/js/app.js"></script>',
        f'  <script>window.FEDFLOPS_CONFIG = {json.dumps(config)};</script>\n'
        '  <script type="module" src="/js/app.js"></script>',
    )
    (OUT / "index.html").write_text(index, encoding="utf-8")

    (OUT / "demo.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    # netlify.toml is ignored by a drag-and-drop deploy, which is how most people
    # publish this. The file-based equivalents (_headers, _redirects) travel with
    # the directory, so both deploy paths behave the same.
    (OUT / "_redirects").write_text(
        "# hash routing needs no rewrite; this only catches stray deep links\n"
        "/*  /index.html  200\n",
        encoding="utf-8",
    )
    (OUT / "_headers").write_text(
        "/*\n"
        "  X-Content-Type-Options: nosniff\n"
        "  X-Frame-Options: DENY\n"
        "  Referrer-Policy: strict-origin-when-cross-origin\n"
        "\n"
        "/js/*\n"
        "  Cache-Control: public, max-age=3600\n"
        "\n"
        "/demo.json\n"
        "  Cache-Control: public, max-age=300\n",
        encoding="utf-8",
    )

    size = (OUT / "demo.json").stat().st_size / 1024
    print(f"\nwrote {OUT}")
    print(f"  demo.json  {size:.0f} KB  ({len(payload['runs'])} runs, "
          f"{len(payload['models'])} versions, {len(payload['dataset_rows'])} rows)")
    print(f"  mode       {'static snapshot (in-browser inference)' if not api_base else f'live API at {api_base}'}")
    print("\ndeploy:  npx netlify deploy --prod --dir netlify/site")
    print("preview: python -m http.server 8080 --directory netlify/site")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the static Netlify site")
    parser.add_argument("--api-base", default="", help="point the site at a live API instead of the snapshot")
    args = parser.parse_args()
    build(api_base=args.api_base.rstrip("/"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
