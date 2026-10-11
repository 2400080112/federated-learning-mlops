
import numpy as np
import pytest

from src.fedmlops.model import (
    evaluate,
    initialize_weights,
    predict_probabilities,
    sigmoid,
    train_local,
)


def test_initialize_weights():
    weights = initialize_weights(3)

    assert len(weights.coefficients) == 3
    assert np.all(weights.coefficients == 0)
    assert weights.intercept == 0.0


def test_sigmoid():
    result = sigmoid(np.array([0.0]))

    assert np.allclose(result, [0.5])


def test_predictions_are_probabilities():
    weights = initialize_weights(2)
    features = np.array([[1.0, 2.0], [3.0, 4.0]])

    probabilities = predict_probabilities(weights, features)

    assert len(probabilities) == 2
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_training_updates_weights():
    weights = initialize_weights(2)
    features = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([1.0, 0.0])

    updated = train_local(
        weights,
        features,
        labels,
        epochs=5,
        learning_rate=0.1,
        l2_strength=0.0,
    )

    assert not np.allclose(
        updated.coefficients,
        weights.coefficients,
    )


def test_evaluate_returns_metrics():
    weights = initialize_weights(2)
    features = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([1.0, 0.0])

    metrics = evaluate(weights, features, labels)

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["loss"] >= 0


def test_evaluate_classification_metrics():
    weights = initialize_weights(1)
    weights.coefficients = np.array([1.0])
    weights.intercept = 0.0

    features = np.array([[2.0], [-2.0], [1.0], [-1.0]])
    labels = np.array([1.0, 0.0, 0.0, 1.0])

    metrics = evaluate(weights, features, labels)

    assert metrics["true_positive"] == 1.0
    assert metrics["true_negative"] == 1.0
    assert metrics["false_positive"] == 1.0
    assert metrics["false_negative"] == 1.0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert np.isclose(metrics["f1_score"], 0.5)


def test_dp_training_returns_finite_weights():
    weights = initialize_weights(2)
    features = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    labels = np.array([1.0, 0.0, 1.0])

    updated = train_local(
        weights,
        features,
        labels,
        epochs=2,
        learning_rate=0.1,
        l2_strength=0.0,
        differential_privacy=True,
        clipping_norm=1.0,
        noise_multiplier=1.0,
        random_generator=np.random.default_rng(42),
    )

    assert np.all(np.isfinite(updated.coefficients))
    assert np.isfinite(updated.intercept)


def test_dp_training_uses_noise():
    weights = initialize_weights(2)
    features = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([1.0, 0.0])

    first = train_local(
        weights,
        features,
        labels,
        epochs=1,
        learning_rate=0.1,
        l2_strength=0.0,
        differential_privacy=True,
        clipping_norm=1.0,
        noise_multiplier=1.0,
        random_generator=np.random.default_rng(1),
    )

    second = train_local(
        weights,
        features,
        labels,
        epochs=1,
        learning_rate=0.1,
        l2_strength=0.0,
        differential_privacy=True,
        clipping_norm=1.0,
        noise_multiplier=1.0,
        random_generator=np.random.default_rng(2),
    )

    assert not np.allclose(first.coefficients, second.coefficients)


def test_dp_invalid_clipping_norm():
    weights = initialize_weights(1)
    features = np.array([[1.0]])
    labels = np.array([1.0])

    with pytest.raises(ValueError):
        train_local(
            weights,
            features,
            labels,
            epochs=1,
            learning_rate=0.1,
            l2_strength=0.0,
            differential_privacy=True,
            clipping_norm=0.0,
        )


def test_dp_invalid_noise_multiplier():
    weights = initialize_weights(1)
    features = np.array([[1.0]])
    labels = np.array([1.0])

    with pytest.raises(ValueError):
        train_local(
            weights,
            features,
            labels,
            epochs=1,
            learning_rate=0.1,
            l2_strength=0.0,
            differential_privacy=True,
            noise_multiplier=-1.0,
        )
