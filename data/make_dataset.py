"""Export the Wisconsin Diagnostic Breast Cancer (WDBC) dataset to CSV.

This script is provenance only: it was used once to produce
``data/breast_cancer.csv``, which ships with the project. The demo itself
never imports scikit-learn at run time -- the CSV is bundled so the project
runs with numpy + pandas only.

Usage (needs scikit-learn, which the project itself does not):
    python data/make_dataset.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from sklearn.datasets import load_breast_cancer

TARGET_NAMES = {0: "malignant", 1: "benign"}


def main() -> None:
    bunch = load_breast_cancer()
    out = Path(__file__).resolve().parent / "breast_cancer.csv"

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_id", *bunch.feature_names, "target", "target_name"])
        for i, (row, label) in enumerate(zip(bunch.data, bunch.target)):
            writer.writerow([i, *[f"{v:.6f}" for v in row], int(label), TARGET_NAMES[int(label)]])

    print(f"wrote {out} rows={len(bunch.target)} features={len(bunch.feature_names)}")


if __name__ == "__main__":
    main()
