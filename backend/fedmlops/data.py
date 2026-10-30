"""Loading and splitting the bundled WDBC dataset.

No scikit-learn at run time: the CSV ships with the project and pandas reads
it. Labels are 1 = benign, 0 = malignant (the original UCI/Wisconsin coding).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # backend/fedmlops/data.py -> repo root


@dataclass
class Dataset:
    """A scaled feature matrix plus labels and the scaler that produced it."""

    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    mean: np.ndarray
    std: np.ndarray

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def subset(self, idx: np.ndarray) -> "Dataset":
        return Dataset(
            X=self.X[idx],
            y=self.y[idx],
            feature_names=self.feature_names,
            mean=self.mean,
            std=self.std,
        )


def resolve_path(path: str | Path) -> Path:
    """Accept an absolute path, a cwd-relative path, or a repo-relative path."""
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return PROJECT_ROOT / p


def load_raw(csv_path: str | Path, target_column: str = "target",
             drop_columns: Tuple[str, ...] = ()) -> Tuple[pd.DataFrame, pd.Series]:
    path = resolve_path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found: {path}\n"
            "The CSV is bundled at data/breast_cancer.csv; run commands from the "
            "project root, or pass --config with an absolute data path."
        )
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"target column {target_column!r} not in {path.name}")
    drop = [c for c in drop_columns if c in df.columns]
    y = df[target_column].astype(int)
    X = df.drop(columns=[target_column, *drop])
    return X, y


def standardize(train: pd.DataFrame, *others: pd.DataFrame) -> Tuple[Dataset, ...]:
    """Fit mean/std on the training frame only, then apply to every split."""
    mean = np.asarray(train.mean(axis=0).to_numpy(dtype=float))
    std = np.array(train.std(axis=0).to_numpy(dtype=float))  # copy: pandas may hand back a read-only view
    std[std == 0] = 1.0

    out: List[Dataset] = []
    for frame in (train, *others):
        arr = (frame.to_numpy(dtype=float) - mean) / std
        out.append(
            Dataset(
                X=arr,
                y=np.zeros(0),  # filled by caller
                feature_names=list(frame.columns),
                mean=mean,
                std=std,
            )
        )
    return tuple(out)


def _stratified_indices(y: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Pick ``frac`` of each class, so both classes appear in every split."""
    chosen: List[np.ndarray] = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        take = max(1, int(round(len(idx) * frac)))
        chosen.append(idx[:take])
    return np.sort(np.concatenate(chosen))


def load_splits(
    csv_path: str | Path,
    target_column: str = "target",
    drop_columns: Tuple[str, ...] = (),
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 7,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Return (train, val, test), scaled with statistics from train only.

    ``val`` is used for per-round monitoring; ``test`` is touched once, at the
    end, so the reported number is not a number we tuned against.
    """
    X_df, y_s = load_raw(csv_path, target_column, drop_columns)
    y = y_s.to_numpy(dtype=int)
    rng = np.random.default_rng(seed)

    test_idx = _stratified_indices(y, test_size, rng)
    mask = np.ones(len(y), dtype=bool)
    mask[test_idx] = False

    remaining = np.flatnonzero(mask)
    val_local = _stratified_indices(y[remaining], val_size / (1.0 - test_size), rng)
    val_idx = remaining[val_local]
    train_idx = np.setdiff1d(remaining, val_idx, assume_unique=False)

    frames = [X_df.iloc[i].reset_index(drop=True) for i in (train_idx, val_idx, test_idx)]
    ds_train, ds_val, ds_test = standardize(*frames)

    for ds, idx in zip((ds_train, ds_val, ds_test), (train_idx, val_idx, test_idx)):
        ds.y = y[idx]

    return ds_train, ds_val, ds_test
