"""Logistic regression trained locally on each client.

Parameters are kept as one flat vector ``[w_0 ... w_d, b]`` so that FedAvg is a
single weighted average and so a model is trivially JSON-serialisable for the
registry. Deliberately plain NumPy: no autograd, no framework, nothing hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


@dataclass
class LogisticRegression:
    n_features: int
    l2: float = 0.0

    def __post_init__(self) -> None:
        self.params = np.zeros(self.n_features + 1, dtype=float)

    # ---- parameter plumbing -------------------------------------------
    @property
    def weights(self) -> np.ndarray:
        return self.params[:-1]

    @property
    def bias(self) -> float:
        return float(self.params[-1])

    def get_params(self) -> np.ndarray:
        return self.params.copy()

    def set_params(self, params: np.ndarray) -> "LogisticRegression":
        params = np.asarray(params, dtype=float)
        if params.shape != (self.n_features + 1,):
            raise ValueError(
                f"expected {self.n_features + 1} parameters, got {params.shape}"
            )
        self.params = params.copy()
        return self

    # ---- inference -----------------------------------------------------
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ---- training ------------------------------------------------------
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(self.predict_proba(X), 1e-12, 1 - 1e-12)
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(bce + 0.5 * self.l2 * float(np.dot(self.weights, self.weights)))

    def local_train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
        learning_rate: float = 0.1,
        batch_size: int = 16,
        rng: np.random.Generator | None = None,
        weights: np.ndarray | None = None,
    ) -> "LogisticRegression":
        """A few epochs of mini-batch SGD, in place.

        ``weights`` (optional) is a per-sample weight vector, used to report an
        unweighted class-balanced loss for skewed clients. Training itself is
        unweighted by default.
        """
        n = len(y)
        if n == 0:
            return self
        rng = rng or np.random.default_rng(0)

        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                Xb, yb = X[idx], y[idx]
                err = self.predict_proba(Xb) - yb
                grad_w = Xb.T @ err / len(idx) + self.l2 * self.weights
                grad_b = float(np.mean(err))
                self.params[:-1] -= learning_rate * grad_w
                self.params[-1] -= learning_rate * grad_b
        return self

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": "logistic_regression",
            "n_features": int(self.n_features),
            "l2": float(self.l2),
            "params": [float(v) for v in self.params],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "LogisticRegression":
        if raw.get("kind") != "logistic_regression":
            raise ValueError(f"unsupported model kind: {raw.get('kind')!r}")
        model = cls(n_features=int(raw["n_features"]), l2=float(raw.get("l2", 0.0)))
        model.set_params(np.asarray(raw["params"], dtype=float))
        return model


def fedavg(
    client_params: List[np.ndarray],
    client_sizes: List[int],
) -> np.ndarray:
    """Size-weighted average of client parameter vectors -- the FedAvg step."""
    if not client_params:
        raise ValueError("fedavg needs at least one client update")
    if len(client_params) != len(client_sizes):
        raise ValueError("client_params and client_sizes must be the same length")
    total = float(sum(client_sizes))
    if total <= 0:
        raise ValueError("total sample count must be positive")
    stacked = np.stack([np.asarray(p, dtype=float) for p in client_params], axis=0)
    weights = np.asarray(client_sizes, dtype=float) / total
    return weights @ stacked
