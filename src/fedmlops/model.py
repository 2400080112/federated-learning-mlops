
"""NumPy logistic regression with optional DP-SGD training."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ModelWeights:
    coefficients: np.ndarray
    intercept: float

    def copy(self) -> "ModelWeights":
        return ModelWeights(
            self.coefficients.copy(),
            float(self.intercept),
        )


def initialize_weights(feature_count: int) -> ModelWeights:
    return ModelWeights(
        np.zeros(feature_count, dtype=np.float64),
        0.0,
    )


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
    *,
    differential_privacy: bool = False,
    clipping_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    random_generator=None,
) -> ModelWeights:
    """Train a local model, optionally using clipped noisy gradients.

    The DP mode clips each example's combined coefficient/intercept
    gradient and adds Gaussian noise to the summed clipped gradients.

    A formal privacy guarantee requires a suitable accountant and
    valid assumptions about sampling, adjacency, and all releases.
    """
    updated = weights.copy()
    row_count = len(features)

    if row_count == 0:
        raise ValueError("Training data must not be empty.")

    if differential_privacy:
        if clipping_norm <= 0:
            raise ValueError("clipping_norm must be positive.")
        if noise_multiplier < 0:
            raise ValueError("noise_multiplier cannot be negative.")

        if random_generator is None:
            # Use NumPy's secure operating-system-backed generator.
            random_generator = np.random.default_rng()

    for _ in range(epochs):
        probabilities = sigmoid(
            features @ updated.coefficients + updated.intercept
        )
        errors = probabilities - labels

        if differential_privacy:
            # Per-example gradients for logistic regression.
            per_example_weights = (
                errors[:, None] * features
            )
            per_example_intercept = errors[:, None]

            # Include the intercept in the clipping norm.
            per_example_gradients = np.concatenate(
                [per_example_weights, per_example_intercept],
                axis=1,
            )

            gradient_norms = np.linalg.norm(
                per_example_gradients,
                axis=1,
            )

            clipping_factors = np.minimum(
                1.0,
                clipping_norm / np.maximum(gradient_norms, 1e-12),
            )

            clipped_gradients = (
                per_example_gradients
                * clipping_factors[:, None]
            )

            # Sum clipped gradients, add Gaussian noise, then average.
            gradient_sum = np.sum(clipped_gradients, axis=0)

            noise = random_generator.normal(
                loc=0.0,
                scale=noise_multiplier * clipping_norm,
                size=gradient_sum.shape,
            )

            noisy_average_gradient = (
                gradient_sum + noise
            ) / row_count

            grad_weights = noisy_average_gradient[:-1]
            grad_intercept = float(noisy_average_gradient[-1])

            # Regularization is applied separately.
            grad_weights = (
                grad_weights
                + l2_strength * updated.coefficients
            )

        else:
            # Preserve the original non-private training behaviour.
            grad_weights = (
                (features.T @ errors) / row_count
                + l2_strength * updated.coefficients
            )
            grad_intercept = float(np.mean(errors))

        updated.coefficients -= learning_rate * grad_weights
        updated.intercept -= learning_rate * grad_intercept

    return updated


def predict_probabilities(
    weights: ModelWeights,
    features: np.ndarray,
) -> np.ndarray:
    return sigmoid(
        features @ weights.coefficients + weights.intercept
    )


def evaluate(
    weights: ModelWeights,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Evaluate loss and binary classification metrics."""
    probabilities = predict_probabilities(weights, features)
    predictions = (probabilities >= 0.5).astype(np.float64)

    eps = 1e-12
    loss = -np.mean(
        labels * np.log(probabilities + eps)
        + (1 - labels) * np.log(1 - probabilities + eps)
    )

    # Positive class: benign (label 1).
    true_positive = int(
        np.sum((predictions == 1) & (labels == 1))
    )
    true_negative = int(
        np.sum((predictions == 0) & (labels == 0))
    )
    false_positive = int(
        np.sum((predictions == 1) & (labels == 0))
    )
    false_negative = int(
        np.sum((predictions == 0) & (labels == 1))
    )

    precision = true_positive / max(
        true_positive + false_positive, 1
    )
    recall = true_positive / max(
        true_positive + false_negative, 1
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "loss": float(loss),
        "accuracy": float(np.mean(predictions == labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "true_positive": float(true_positive),
        "true_negative": float(true_negative),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }
