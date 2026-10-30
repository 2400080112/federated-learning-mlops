"""Terminal interface for the same pipeline the web app drives.

    python -m app.cli train --clients 8 --rounds 25
    python -m app.cli train --partition dirichlet --alpha 0.3 --dp-epsilon 8
    python -m app.cli evaluate --run <run-id>
    python -m app.cli register --run <run-id>
    python -m app.cli promote --version-id 1 --min-accuracy 0.9
    python -m app.cli predict --limit 100
    python -m app.cli list-runs

Same database, same services, same rules as the HTTP API -- this is the
scriptable surface for experiments and CI, not a second implementation. Training
here runs synchronously and prints each round, which is what you want in a
terminal (the API queues it to a worker instead).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .db import init_db, make_engine, make_session_factory
from .models import ModelVersion, Run
from .schemas import RunCreate
from .services import prediction_service, registry_service, training


def _session_factory():
    engine = make_engine()
    init_db(engine)
    return make_session_factory(engine)


def _print_run(run: Run) -> None:
    print(f"  run        {run.id}  ({run.status})")
    if run.val_accuracy is not None:
        print(f"  val        acc={run.val_accuracy:.4f}  f1={run.val_f1:.4f}  auc={run.val_roc_auc:.4f}")
    if run.test_accuracy is not None:
        print(f"  test       acc={run.test_accuracy:.4f}  f1={run.test_f1:.4f}  auc={run.test_roc_auc:.4f}")
    if run.privacy_enabled:
        print(f"  privacy    eps={run.epsilon:.2f}  sigma={run.noise_multiplier:.3f}  clip={run.clip_norm}")


# ------------------------------------------------------------------ commands
def cmd_train(args: argparse.Namespace) -> int:
    payload = RunCreate(
        name=args.name or "",
        n_clients=args.clients,
        clients_per_round=args.clients_per_round,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        partition=args.partition,
        dirichlet_alpha=args.alpha,
        seed=args.seed,
        privacy_enabled=bool(args.dp_epsilon or args.dp_sigma),
        dp_epsilon=args.dp_epsilon,
        dp_sigma=args.dp_sigma,
        dp_clip=args.dp_clip,
    )
    factory = _session_factory()
    session = factory()
    try:
        run = training.create_run(session, payload)
        print(f"training {run.name} ({run.id})")
        print(f"  {args.clients} clients, {args.partition}, {args.rounds} rounds")

        def progress(record) -> None:
            line = (
                f"  round {record.round:>3}  loss={record.train_loss:.4f}  "
                f"val_acc={record.val_accuracy:.4f}  val_auc={record.val_roc_auc:.4f}"
            )
            if record.epsilon:
                line += f"  eps={record.epsilon:.2f}"
            if record.clipped_fraction:
                line += f"  clipped={record.clipped_fraction:.0%}"
            print(line, flush=True)

        training.run_training(factory, run.id, on_round=progress)
        session.expire_all()
        run = session.get(Run, run.id)
        print()
        _print_run(run)
        if run.status != "finished":
            print(f"  error      {run.error}", file=sys.stderr)
            return 1
        print(f"\nnext: python -m app.cli evaluate --run {run.id}")
        return 0
    finally:
        session.close()


def cmd_evaluate(args: argparse.Namespace) -> int:
    factory = _session_factory()
    session = factory()
    try:
        run = session.get(Run, args.run)
        if run is None:
            print(f"no such run: {args.run}", file=sys.stderr)
            return 2
        scores = registry_service.evaluate_run(session, run)
        print(f"test split ({scores['n']} rows)")
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"):
            print(f"  {key:<10} {scores[key]:.4f}")
        return 0
    finally:
        session.close()


def cmd_register(args: argparse.Namespace) -> int:
    factory = _session_factory()
    session = factory()
    try:
        run = session.get(Run, args.run)
        if run is None:
            print(f"no such run: {args.run}", file=sys.stderr)
            return 2
        version = registry_service.register_run(
            session, run, name=args.model, stage=args.stage, description=args.description
        )
        print(f"registered {version.name}:{version.version} ({version.stage})")
        print(f"  test accuracy {version.test_accuracy:.4f}")
        print(f"\nnext: python -m app.cli promote --version-id {version.id}")
        return 0
    finally:
        session.close()


def cmd_promote(args: argparse.Namespace) -> int:
    factory = _session_factory()
    session = factory()
    try:
        version = session.get(ModelVersion, args.version_id)
        if version is None:
            print(f"no such model version: {args.version_id}", file=sys.stderr)
            return 2
        registry_service.promote(
            session, version, stage=args.stage, min_accuracy=args.min_accuracy
        )
        print(f"{version.name}:{version.version} -> {version.stage}")
        return 0
    finally:
        session.close()


def cmd_predict(args: argparse.Namespace) -> int:
    factory = _session_factory()
    session = factory()
    try:
        if args.version_id:
            version = session.get(ModelVersion, args.version_id)
            if version is None:
                print(f"no such model version: {args.version_id}", file=sys.stderr)
                return 2
        else:
            version = registry_service.production_model(session, args.model)
            if version is None:
                print(f"no version of {args.model} in production", file=sys.stderr)
                return 2

        csv_bytes = None
        if args.csv:
            with open(args.csv, "rb") as handle:
                csv_bytes = handle.read()

        record = prediction_service.predict(
            session,
            version,
            csv_bytes=csv_bytes,
            use_bundled=not args.csv,
            threshold=args.threshold,
            limit=args.limit,
        )
        summary = json.loads(record.summary_json)
        print(f"{record.model_label} scored {record.n_rows} rows ({record.source_label})")
        print(f"  benign     {summary['predicted_benign']}")
        print(f"  malignant  {summary['predicted_malignant']}")
        if "accuracy" in summary:
            print(f"  accuracy   {summary['accuracy']:.4f} (vs the input's target column)")
        print(f"  csv        {record.output_path}")
        return 0
    finally:
        session.close()


def cmd_list_runs(args: argparse.Namespace) -> int:
    factory = _session_factory()
    session = factory()
    try:
        from sqlalchemy import select

        runs = session.scalars(
            select(Run).order_by(Run.created_at.desc()).limit(args.limit)
        ).all()
        if not runs:
            print("no runs yet")
            return 0
        print(f"{'run id':<30} {'status':<9} {'clients':>7} {'val':>7} {'test':>7}  name")
        for run in runs:
            val = f"{run.val_accuracy:.4f}" if run.val_accuracy is not None else "-"
            test = f"{run.test_accuracy:.4f}" if run.test_accuracy is not None else "-"
            clients = json.loads(run.config_json or "{}").get("federated", {}).get("n_clients", "-")
            print(f"{run.id:<30} {run.status:<9} {clients:>7} {val:>7} {test:>7}  {run.name}")
        return 0
    finally:
        session.close()


def cmd_list_models(args: argparse.Namespace) -> int:
    factory = _session_factory()
    session = factory()
    try:
        from sqlalchemy import select

        versions = session.scalars(
            select(ModelVersion).order_by(ModelVersion.id.desc())
        ).all()
        if not versions:
            print("registry is empty")
            return 0
        print(f"{'id':>3}  {'version':<22} {'stage':<11} {'test acc':>9}  source run")
        for v in versions:
            acc = f"{v.test_accuracy:.4f}" if v.test_accuracy is not None else "-"
            print(f"{v.id:>3}  {v.name + ':' + v.version:<22} {v.stage:<11} {acc:>9}  {v.run_id or '-'}")
        return 0
    finally:
        session.close()


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Federated learning with MLOps -- terminal interface.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a federated run")
    p.add_argument("--clients", type=int, default=8)
    p.add_argument("--clients-per-round", type=int, default=None)
    p.add_argument("--rounds", type=int, default=25)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=0.15)
    p.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dp-epsilon", type=float, default=None)
    p.add_argument("--dp-sigma", type=float, default=None)
    p.add_argument("--dp-clip", type=float, default=1.0)
    p.add_argument("--name", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="score a run on the test split")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("register", help="register a scored run")
    p.add_argument("--run", required=True)
    p.add_argument("--model", default="wdbc-logreg")
    p.add_argument("--stage", default="staging", choices=["staging", "production", "archived"])
    p.add_argument("--description", default="")
    p.set_defaults(func=cmd_register)

    p = sub.add_parser("promote", help="move a version into a stage")
    p.add_argument("--version-id", type=int, required=True)
    p.add_argument("--stage", default="production", choices=["staging", "production", "archived"])
    p.add_argument("--min-accuracy", type=float, default=None)
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("predict", help="score a CSV with a registered model")
    p.add_argument("--csv", default=None, help="default: the bundled dataset")
    p.add_argument("--model", default="wdbc-logreg")
    p.add_argument("--version-id", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("list-runs", help="list tracked runs")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_list_runs)

    p = sub.add_parser("list-models", help="list registry versions")
    p.set_defaults(func=cmd_list_models)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (registry_service.ServiceError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
