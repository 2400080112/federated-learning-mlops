
"""Federated training loop and MLOps run tracking."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import load_federated_data
from .model import evaluate, initialize_weights, train_local
from .privacy import calculate_epsilon
from .registry import save_model


def run_pipeline(
    config: dict,
    num_clients: int | None = None,
    rounds: int | None = None,
) -> Path:
    """Run federated training and save experiment artifacts."""

    effective = dict(config)

    if num_clients is not None:
        effective["num_clients"] = num_clients

    if rounds is not None:
        effective["rounds"] = rounds

    # Read training and privacy settings.
    dp_enabled = effective.get("differential_privacy", False)
    clipping_norm = effective.get("clipping_norm", 1.0)
    noise_multiplier = effective.get("noise_multiplier", 1.0)
    privacy_delta = effective.get("privacy_delta", 1e-5)

    # Validate configuration.
    if effective["num_clients"] <= 0:
        raise ValueError("num_clients must be positive.")

    if effective["rounds"] <= 0:
        raise ValueError("rounds must be positive.")

    if effective["local_epochs"] <= 0:
        raise ValueError("local_epochs must be positive.")

    if dp_enabled:
        if clipping_norm <= 0:
            raise ValueError("clipping_norm must be positive.")

        if noise_multiplier <= 0:
            raise ValueError(
                "noise_multiplier must be positive for DP training."
            )

        if not 0 < privacy_delta < 1:
            raise ValueError("privacy_delta must be between 0 and 1.")

    # Load and distribute the dataset among clients.
    data = load_federated_data(
        effective["num_clients"],
        effective["test_size"],
        effective["random_seed"],
    )

    # Initialize the global model.
    global_weights = initialize_weights(
        data.client_features[0].shape[1]
    )

    history: list[dict] = []
    communication_history: list[dict] = []

    # Estimate the numeric payload size of model parameters.
    num_parameters = len(global_weights.coefficients) + 1
    bytes_per_parameter = global_weights.coefficients.dtype.itemsize
    update_size_bytes = num_parameters * bytes_per_parameter

    # Federated training rounds.
    for round_number in range(1, effective["rounds"] + 1):
        client_updates = []

        for x_client, y_client in zip(
            data.client_features,
            data.client_labels,
        ):
            local_weights = train_local(
                global_weights,
                x_client,
                y_client,
                effective["local_epochs"],
                effective["learning_rate"],
                effective["l2_strength"],
                differential_privacy=dp_enabled,
                clipping_norm=clipping_norm,
                noise_multiplier=noise_multiplier,
            )

            client_updates.append(
                (local_weights, len(x_client))
            )

        # Estimate upload and download payload sizes.
        number_of_clients = len(client_updates)

        upload_bytes = number_of_clients * update_size_bytes
        download_bytes = number_of_clients * update_size_bytes

        communication_history.append({
            "round": round_number,
            "clients": number_of_clients,
            "parameters_per_update": num_parameters,
            "upload_bytes": upload_bytes,
            "download_bytes": download_bytes,
            "total_bytes": upload_bytes + download_bytes,
        })

        # Aggregate client models using sample-count-weighted FedAvg.
        total_examples = sum(
            count for _, count in client_updates
        )

        global_weights.coefficients = sum(
            weights.coefficients * count
            for weights, count in client_updates
        ) / total_examples

        global_weights.intercept = sum(
            weights.intercept * count
            for weights, count in client_updates
        ) / total_examples

        # Evaluate the global model.
        metrics = evaluate(
            global_weights,
            data.test_features,
            data.test_labels,
        )

        history.append({
            "round": round_number,
            **metrics,
        })

        print(
            f"Round {round_number:02d}/{effective['rounds']}: "
            f"loss={metrics['loss']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}"
        )

    final_metrics = history[-1]

    # Calculate the privacy estimate when DP is enabled.
    privacy_report = None

    if dp_enabled:
        accounted_steps = (
            effective["rounds"] * effective["local_epochs"]
        )

        epsilon = calculate_epsilon(
            noise_multiplier=noise_multiplier,
            steps=accounted_steps,
            delta=privacy_delta,
        )

        privacy_report = {
            "accounting_method": (
                "RDP with full-batch Gaussian mechanism"
            ),
            "epsilon": epsilon,
            "delta": privacy_delta,
            "noise_multiplier": noise_multiplier,
            "clipping_norm": clipping_norm,
            "accounted_steps_per_client": accounted_steps,
            "adjacency": "replace-one example",
            "status": (
                "Theoretical estimate under documented assumptions"
            ),
            "limitations": [
                (
                    "This estimate assumes per-example gradients are "
                    "clipped and Gaussian noise is added to their sum."
                ),
                (
                    "The privacy accountant must match the exact "
                    "training mechanism and threat model."
                ),
                (
                    "Data-dependent preprocessing, including "
                    "standardization, is not privatized by this mechanism."
                ),
                (
                    "This report alone does not establish an "
                    "end-to-end differential privacy guarantee."
                ),
            ],
        }

    # Create a unique run directory.
    run_id_timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    project_root = Path(__file__).resolve().parents[2]

    run_dir = (
        project_root
        / "artifacts"
        / "runs"
        / run_id_timestamp
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save effective configuration.
    (run_dir / "config.json").write_text(
        json.dumps(effective, indent=2),
        encoding="utf-8",
    )

    # Save privacy accounting results when DP is enabled.
    if privacy_report is not None:
        (run_dir / "privacy.json").write_text(
            json.dumps(privacy_report, indent=2),
            encoding="utf-8",
        )

    # Save metrics for each training round.
    with (run_dir / "metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "round",
                "loss",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "true_positive",
                "true_negative",
                "false_positive",
                "false_negative",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    # Save estimated communication for each round.
    with (run_dir / "communication.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as communication_file:
        writer = csv.DictWriter(
            communication_file,
            fieldnames=[
                "round",
                "clients",
                "parameters_per_update",
                "upload_bytes",
                "download_bytes",
                "total_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(communication_history)

    # Register the final global model.
    version_dir = save_model(
        global_weights,
        final_metrics,
        effective,
        data.feature_names,
        data.scaler_mean,
        data.scaler_scale,
    )

    total_communication_bytes = sum(
        item["total_bytes"]
        for item in communication_history
    )

    # Print experiment summary.
    mode = "DP-FedAvg" if dp_enabled else "Standard FedAvg"

    print(f"\nTraining mode: {mode}")

    if dp_enabled and privacy_report is not None:
        print(f"Clipping norm: {clipping_norm}")
        print(f"Noise multiplier: {noise_multiplier}")
        print(
            f"Estimated epsilon: {privacy_report['epsilon']:.4f}"
        )
        print(f"Privacy delta: {privacy_report['delta']}")
        print(
            "Note: privacy estimate is theoretical; "
            "review privacy.json for assumptions."
        )

    print(f"Final accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Registered model: {version_dir}")
    print(
        "Estimated total communication: "
        f"{total_communication_bytes:,} bytes"
    )
    print(f"Communication log: {run_dir / 'communication.csv'}")

    if privacy_report is not None:
        print(f"Privacy report: {run_dir / 'privacy.json'}")

    return version_dir
