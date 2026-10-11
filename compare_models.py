
"""Compare centralized training with federated FedAvg training."""

import json
from pathlib import Path

import pandas as pd

from src.fedmlops.data import load_federated_data
from src.fedmlops.model import evaluate, initialize_weights, train_local


ROOT = Path(__file__).resolve().parent


def main():
    config = json.loads(
        (ROOT / "config.json").read_text(encoding="utf-8")
    )

    data = load_federated_data(
        num_clients=config["num_clients"],
        test_size=config["test_size"],
        seed=config["random_seed"],
    )

    # Reconstruct the complete training set from client partitions.
    x_train = pd.concat(
        [
            pd.DataFrame(x)
            for x in data.client_features
        ],
        ignore_index=True,
    ).to_numpy()

    y_train = pd.concat(
        [
            pd.Series(y)
            for y in data.client_labels
        ],
        ignore_index=True,
    ).to_numpy()

    # 1. Centralized model: all training data in one training process.
    centralized_weights = initialize_weights(x_train.shape[1])

    centralized_weights = train_local(
        centralized_weights,
        x_train,
        y_train,
        epochs=config["rounds"] * config["local_epochs"],
        learning_rate=config["learning_rate"],
        l2_strength=config["l2_strength"],
    )

    centralized_metrics = evaluate(
        centralized_weights,
        data.test_features,
        data.test_labels,
    )

    # 2. Federated model: local training and sample-weighted FedAvg.
    federated_weights = initialize_weights(
        data.client_features[0].shape[1]
    )

    for _ in range(config["rounds"]):
        updates = []

        for x_client, y_client in zip(
            data.client_features,
            data.client_labels,
        ):
            local_weights = train_local(
                federated_weights,
                x_client,
                y_client,
                epochs=config["local_epochs"],
                learning_rate=config["learning_rate"],
                l2_strength=config["l2_strength"],
            )
            updates.append((local_weights, len(x_client)))

        total_samples = sum(count for _, count in updates)

        # Aggregate coefficients and intercept by client sample count.
        federated_weights.coefficients = sum(
            weights.coefficients * count
            for weights, count in updates
        ) / total_samples

        federated_weights.intercept = sum(
            weights.intercept * count
            for weights, count in updates
        ) / total_samples

    federated_metrics = evaluate(
        federated_weights,
        data.test_features,
        data.test_labels,
    )

    # 3. Save measured comparison results.
    comparison = pd.DataFrame(
        [
            {"model": "Centralized", **centralized_metrics},
            {"model": "Federated FedAvg", **federated_metrics},
        ]
    )

    output_dir = ROOT / "artifacts" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "comparison.csv"
    comparison.to_csv(output_file, index=False)

    display_columns = [
        "model",
        "accuracy",
        "loss",
        "precision",
        "recall",
        "f1_score",
    ]

    print("\nCENTRALIZED VS FEDERATED RESULTS")
    print(comparison[display_columns].to_string(index=False))
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
