"""A tiny NumPy logistic regression model for transparent FedAvg updates."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ModelWeights:
    coefficients: np.ndarray
    intercept: float

    def copy(self) -> "ModelWeights":
        return ModelWeights(self.coefficients.copy(), float(self.intercept))


def initialize_weights(feature_count: int) -> ModelWeights:
    return ModelWeights(np.zeros(feature_count, dtype=np.float64), 0.0)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-logits))


def train_local(
    weights: ModelWeights,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2_strength: float,
) -> ModelWeights:
    """Train a local copy; client rows are never part of the returned update."""
    updated = weights.copy()
    row_count = len(features)
    for _ in range(epochs):
        probabilities = sigmoid(features @ updated.coefficients + updated.intercept)
        error = probabilities - labels
        grad_weights = (features.T @ error) / row_count + l2_strength * updated.coefficients
        grad_intercept = float(np.mean(error))
        updated.coefficients -= learning_rate * grad_weights
        updated.intercept -= learning_rate * grad_intercept
    return updated


def predict_probabilities(weights: ModelWeights, features: np.ndarray) -> np.ndarray:
    return sigmoid(features @ weights.coefficients + weights.intercept)


def evaluate(weights: ModelWeights, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = predict_probabilities(weights, features)
    predictions = (probabilities >= 0.5).astype(np.float64)
    eps = 1e-12
    loss = -np.mean(labels * np.log(probabilities + eps) + (1 - labels) * np.log(1 - probabilities + eps))
    return {
        "loss": float(loss),
        "accuracy": float(np.mean(predictions == labels)),
    }
