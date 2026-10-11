
import pytest

from src.fedmlops.privacy import calculate_epsilon


def test_full_batch_accounting_returns_positive_epsilon():
    epsilon = calculate_epsilon(
        noise_multiplier=1.0,
        steps=24,
        delta=1e-5,
    )

    assert epsilon > 0


def test_more_steps_increase_epsilon():
    epsilon_10 = calculate_epsilon(1.0, 10, 1e-5)
    epsilon_20 = calculate_epsilon(1.0, 20, 1e-5)

    assert epsilon_20 > epsilon_10


def test_more_noise_reduces_epsilon():
    low_noise = calculate_epsilon(0.5, 10, 1e-5)
    high_noise = calculate_epsilon(2.0, 10, 1e-5)

    assert high_noise < low_noise


def test_accounting_rejects_invalid_noise():
    with pytest.raises(ValueError):
        calculate_epsilon(0, 10, 1e-5)


def test_accounting_rejects_invalid_steps():
    with pytest.raises(ValueError):
        calculate_epsilon(1.0, 0, 1e-5)


def test_accounting_rejects_invalid_delta():
    with pytest.raises(ValueError):
        calculate_epsilon(1.0, 10, 1.0)
