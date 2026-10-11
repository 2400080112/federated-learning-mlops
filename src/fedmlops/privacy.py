
"""RDP accounting for full-batch Gaussian training mechanisms.

This module assumes:
- Per-example gradients are clipped to clipping_norm.
- Gaussian noise is added to the summed clipped gradients.
- Training uses the complete client dataset at every local step.
- Adjacency means replacing one example in a fixed-size client dataset.

For replace-one adjacency, changing one clipped gradient can change
the summed gradient by up to 2 * clipping_norm. Therefore, the
effective noise multiplier for accounting is noise_multiplier / 2.

The result is an upper-bound estimate only when the implementation
and threat model match these assumptions.
"""

from opacus.accountants import RDPAccountant


def calculate_epsilon(
    noise_multiplier: float,
    steps: int,
    delta: float,
) -> float:
    """Account for repeated full-batch Gaussian mechanisms.

    Args:
        noise_multiplier: Standard deviation of added noise divided
            by the clipping norm.
        steps: Number of local noisy-gradient steps.
        delta: Target delta for approximate differential privacy.

    Returns:
        Epsilon under the stated replace-one adjacency assumptions.
    """
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive.")

    if steps <= 0:
        raise ValueError("steps must be positive.")

    if not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1.")

    accountant = RDPAccountant()

    # Replace-one adjacency has sensitivity up to 2*C rather than C.
    effective_noise_multiplier = noise_multiplier / 2.0

    for _ in range(steps):
        accountant.step(
            noise_multiplier=effective_noise_multiplier,
            sample_rate=1.0,
        )

    return float(accountant.get_epsilon(delta=delta))
