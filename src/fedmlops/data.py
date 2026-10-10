"""Dataset loading and deterministic client partitioning (NumPy only)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension",
]


@dataclass
class FederatedData:
    client_features: list[np.ndarray]
    client_labels: list[np.ndarray]
    test_features: np.ndarray
    test_labels: np.ndarray
    feature_names: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


def _stratified_split(labels: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic stratified split without scikit-learn."""
    rng = np.random.RandomState(seed)
    test_total = int(np.ceil(len(labels) * test_size))
    classes, class_counts = np.unique(labels, return_counts=True)
    expected = class_counts * (test_total / len(labels))
    allocation = np.floor(expected).astype(int)
    remainder = test_total - int(allocation.sum())
    if remainder:
        fractions = expected - allocation
        tie_break = rng.random_sample(len(classes))
        order = np.lexsort((tie_break, fractions))
        allocation[order[-remainder:]] += 1

    train_parts = []
    test_parts = []
    for class_label, test_count in zip(classes, allocation):
        indices = np.flatnonzero(labels == class_label)
        rng.shuffle(indices)
        test_parts.append(indices[:test_count])
        train_parts.append(indices[test_count:])
    train_indices = rng.permutation(np.concatenate(train_parts))
    test_indices = rng.permutation(np.concatenate(test_parts))
    return train_indices, test_indices


def load_federated_data(num_clients: int, test_size: float, seed: int) -> FederatedData:
    """Load bundled public data, standardize it, and split training rows among clients."""
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    dataset_path = Path(__file__).with_name("breast_cancer.csv")
    # The bundled CSV has a metadata header followed by 30 features and a 0/1 target.
    raw = np.loadtxt(dataset_path, delimiter=",", skiprows=1, dtype=np.float64)
    features, labels = raw[:, :-1], raw[:, -1]
    train_indices, test_indices = _stratified_split(labels, test_size, seed)
    x_train, x_test = features[train_indices], features[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]

    # Fit preprocessing on training data only to avoid test-set leakage.
    scaler_mean = x_train.mean(axis=0)
    scaler_scale = x_train.std(axis=0)
    scaler_scale[scaler_scale == 0] = 1.0
    x_train = ((x_train - scaler_mean) / scaler_scale).astype(np.float64)
    x_test = ((x_test - scaler_mean) / scaler_scale).astype(np.float64)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train))
    splits = np.array_split(indices, num_clients)
    return FederatedData(
        client_features=[x_train[idx] for idx in splits],
        client_labels=[y_train[idx] for idx in splits],
        test_features=x_test,
        test_labels=y_test,
        feature_names=FEATURE_NAMES,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
    )
