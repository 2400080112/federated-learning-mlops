
"""Compare standard FedAvg and DP-FedAvg experiment runs."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "artifacts" / "runs"
OUTPUT_DIR = ROOT / "artifacts" / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "experiment_comparison.csv"


def read_final_metrics(run_dir: Path) -> dict:
    """Read the final row from a run's metrics.csv."""
    metrics_file = run_dir / "metrics.csv"

    if not metrics_file.exists():
        raise FileNotFoundError(f"Missing file: {metrics_file}")

    with metrics_file.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"No metrics found in {metrics_file}")

    return rows[-1]


def read_json(path: Path) -> dict:
    """Read a JSON file if it exists."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    results = []

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        metrics_file = run_dir / "metrics.csv"
        config_file = run_dir / "config.json"

        # Ignore folders that are not complete experiment runs.
        if not metrics_file.exists() or not config_file.exists():
            continue

        config = read_json(config_file)
        metrics = read_final_metrics(run_dir)
        privacy = read_json(run_dir / "privacy.json")

        dp_enabled = config.get("differential_privacy", False)

        results.append({
            "run_id": run_dir.name,
            "mode": "DP-FedAvg" if dp_enabled else "Standard FedAvg",
            "num_clients": config.get("num_clients", ""),
            "rounds": config.get("rounds", ""),
            "noise_multiplier": (
                config.get("noise_multiplier", "")
                if dp_enabled else ""
            ),
            "accuracy": metrics.get("accuracy", ""),
            "loss": metrics.get("loss", ""),
            "precision": metrics.get("precision", ""),
            "recall": metrics.get("recall", ""),
            "f1_score": metrics.get("f1_score", ""),
            "epsilon": privacy.get("epsilon", ""),
            "delta": privacy.get("delta", ""),
        })

    if not results:
        print("No experiment runs with metrics.csv and config.json found.")
        return

    fieldnames = list(results[0].keys())

    with RESULTS_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Display the three latest runs for a quick comparison.
    latest_runs = sorted(
        results, key=lambda row: row["run_id"]
    )[-3:]

    print("\nLatest three experiment runs")
    print("-" * 100)

    for row in latest_runs:
        print(
            f"{row['run_id']} | {row['mode']} | "
            f"Accuracy: {float(row['accuracy']):.4f} | "
            f"Loss: {float(row['loss']):.4f} | "
            f"Epsilon: {row['epsilon'] or 'N/A'}"
        )

    print(f"\nComparison report saved to:\n{RESULTS_FILE}")


if __name__ == "__main__":
    main()
