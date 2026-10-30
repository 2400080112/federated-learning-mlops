"""Differential privacy for the client updates.

What is actually implemented: the standard **central-DP-style Gaussian
mechanism over client deltas**. Each client's update ``delta = local - global``
is clipped to an L2 norm of ``clip_norm`` (bounds sensitivity), Gaussian noise
``N(0, (sigma * C)^2)`` is added per coordinate, and the noisy deltas are
averaged. Clipping first is what makes the noise scale meaningful.

The epsilon reported is the **basic Gaussian-mechanism composition bound** over
rounds, not a tight accountant (no RDP, no subsampling amplification):

    eps_total ~= sqrt(2 * k * ln(1/delta)) / sigma     (k = number of rounds)

That is honest but loose, and it is the main thing a real deployment would
replace. Read ``docs/PRIVACY.md`` for what this demo does and does not claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def clip_update(delta: np.ndarray, clip_norm: float) -> Tuple[np.ndarray, float]:
    """Scale ``delta`` so its L2 norm is at most ``clip_norm``.

    Returns the clipped vector and the norm it had before clipping, which is
    the useful diagnostic: if the raw norm is routinely far above C, the model
    is being squeezed and accuracy will suffer.
    """
    norm = float(np.linalg.norm(delta))
    if norm <= clip_norm or norm == 0.0:
        return delta, norm
    return delta * (clip_norm / norm), norm


def add_gaussian_noise(
    delta: np.ndarray,
    sigma: float,
    clip_norm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add N(0, (sigma * clip_norm)^2) independently to every coordinate."""
    if sigma <= 0:
        return delta
    noise = rng.normal(loc=0.0, scale=sigma * clip_norm, size=delta.shape)
    return delta + noise


def epsilon_for_rounds(sigma: float, rounds: int, delta: float = 1e-5) -> float:
    """Loose composition bound for k applications of the Gaussian mechanism."""
    if rounds <= 0:
        return 0.0
    if sigma <= 0:
        return float("inf")
    return float(np.sqrt(2.0 * rounds * np.log(1.0 / delta)) / sigma)


def sigma_for_epsilon(epsilon: float, rounds: int, delta: float = 1e-5) -> float:
    """Inverse of :func:`epsilon_for_rounds` -- handy when you fix a budget."""
    if epsilon <= 0 or rounds <= 0:
        raise ValueError("epsilon and rounds must be > 0")
    return float(np.sqrt(2.0 * rounds * np.log(1.0 / delta)) / epsilon)


@dataclass
class PrivacyAccountant:
    """Tracks cumulative epsilon across rounds and can enforce a budget."""

    sigma: float
    delta: float = 1e-5
    max_epsilon: float = 0.0
    rounds_done: int = 0

    @property
    def epsilon(self) -> float:
        return epsilon_for_rounds(self.sigma, self.rounds_done, self.delta)

    def spend_round(self) -> float:
        self.rounds_done += 1
        return self.epsilon

    @property
    def budget_exceeded(self) -> bool:
        return self.max_epsilon > 0 and self.epsilon > self.max_epsilon
