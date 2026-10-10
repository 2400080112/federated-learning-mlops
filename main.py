"""Command-line entry point. Run this file directly from PyCharm."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.fedmlops.model import predict_probabilities
from src.fedmlops.pipeline import run_pipeline
from src.fedmlops.registry import load_latest_model


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated learning MLOps demo")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="run federated training")
    train_parser.add_argument("--clients", type=int, help="number of simulated clients")
    train_parser.add_argument("--rounds", type=int, help="number of aggregation rounds")

    predict_parser = subparsers.add_parser("predict", help="predict using the latest registered model")
    predict_parser.add_argument("--features", type=float, nargs=30, required=True, help="30 raw dataset features")

    args = parser.parse_args()
    if args.command is None:
        args.command = "train"
        args.clients = None
        args.rounds = None

    if args.command == "train":
        config = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))
        run_pipeline(config, args.clients, args.rounds)
    else:
        weights, metadata = load_latest_model()
        raw = np.asarray(args.features, dtype=np.float64).reshape(1, -1)
        scaler = metadata["scaler"]
        standardized = (raw - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
        probability = float(predict_probabilities(weights, standardized)[0])
        label = "benign" if probability >= 0.5 else "malignant"
        print(f"Model version: {metadata['run_id']}")
        print(f"Prediction: {label} (benign probability={probability:.4f})")


if __name__ == "__main__":
    main()
