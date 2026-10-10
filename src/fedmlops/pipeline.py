"""Federated training loop and MLOps run tracking."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import load_federated_data
from .model import evaluate, initialize_weights, train_local
from .registry import save_model


def run_pipeline(config: dict, num_clients: int | None = None, rounds: int | None = None) -> Path:
    effective = dict(config)
    if num_clients is not None:
        effective["num_clients"] = num_clients
    if rounds is not None:
        effective["rounds"] = rounds
    data = load_federated_data(effective["num_clients"], effective["test_size"], effective["random_seed"])
    global_weights = initialize_weights(data.client_features[0].shape[1])
    history: list[dict] = []

    for round_number in range(1, effective["rounds"] + 1):
        client_updates = []
        for x_client, y_client in zip(data.client_features, data.client_labels):
            local_weights = train_local(
                global_weights,
                x_client,
                y_client,
                effective["local_epochs"],
                effective["learning_rate"],
                effective["l2_strength"],
            )
            # In a real system only the update and sample count would be sent.
            client_updates.append((local_weights, len(x_client)))

        total_examples = sum(count for _, count in client_updates)
        global_weights.coefficients = sum(
            weights.coefficients * count for weights, count in client_updates
        ) / total_examples
        global_weights.intercept = sum(
            weights.intercept * count for weights, count in client_updates
        ) / total_examples

        metrics = evaluate(global_weights, data.test_features, data.test_labels)
        history.append({"round": round_number, **metrics})
        print(f"Round {round_number:02d}/{effective['rounds']}: loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}")

    final_metrics = history[-1]
    run_id_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    project_root = Path(__file__).resolve().parents[2]
    run_dir = project_root / "artifacts" / "runs" / run_id_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(effective, indent=2), encoding="utf-8")
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=["round", "loss", "accuracy"])
        writer.writeheader()
        writer.writerows(history)
    version_dir = save_model(
        global_weights,
        final_metrics,
        effective,
        data.feature_names,
        data.scaler_mean,
        data.scaler_scale,
    )
    print(f"\nFinal accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Registered model: {version_dir}")
    return version_dir
