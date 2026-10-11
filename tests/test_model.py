
import numpy as np

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
        updated.coefficients, weights.coefficients
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
