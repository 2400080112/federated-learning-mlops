"""Split a training set across simulated clients.

Two regimes, both worth showing in a demo:

* ``iid``      -- shuffle and cut into equal chunks. Every client sees roughly
                  the same class balance, so FedAvg behaves almost like
                  centralised training.
* ``dirichlet``-- draw each client's class proportions from a Dirichlet
                  distribution. Low alpha means clients are skewed (one client
                  may hold almost only malignant samples), which is what real
                  hospitals look like and what makes federated learning hard.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .data import Dataset


def iid_partition(ds: Dataset, n_clients: int, seed: int = 0) -> List[Dataset]:
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1")
    if n_clients > len(ds):
        raise ValueError(
            f"cannot give {n_clients} clients a share of {len(ds)} samples"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))
    return [ds.subset(np.sort(chunk)) for chunk in np.array_split(order, n_clients)]


def _allocate(counts: np.ndarray, total: int) -> np.ndarray:
    """Turn fractional counts into integers that sum to exactly ``total``.

    Largest-remainder rounding: floors everything, then hands the leftover
    samples to the clients with the biggest fractional part.
    """
    counts = np.maximum(np.asarray(counts, dtype=float), 0.0)
    if counts.sum() <= 0:
        counts = np.ones_like(counts)
    counts = counts / counts.sum() * total
    ints = np.floor(counts).astype(int)
    remainder = total - int(ints.sum())
    if remainder > 0:
        order = np.argsort(-(counts - ints))
        ints[order[:remainder]] += 1
    return ints


def dirichlet_partition(
    ds: Dataset,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 0,
    min_samples: int = 5,
    max_attempts: int = 200,
) -> List[Dataset]:
    """Label-skewed partition via a Dirichlet draw over class proportions.

    Each class is dealt out to the clients in proportions drawn from
    ``Dirichlet(alpha)``, so one client can end up with almost only malignant
    samples -- which is the interesting, hard case for FedAvg.

    Two things stop it degenerating: a per-class floor so no client is starved
    to nothing (which would make it a no-op client and break the size-weighted
    average), and a retry loop in case a draw still lands below
    ``min_samples`` overall.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1")

    rng = np.random.default_rng(seed)
    labels = np.unique(ds.y)
    class_idx = {lab: rng.permutation(np.flatnonzero(ds.y == lab)) for lab in labels}

    # give every client at least this many samples of every class it can hold
    floor = max(1, int(np.ceil(min_samples / max(1, len(labels)))))

    for _ in range(max_attempts):
        proportions = rng.dirichlet(np.repeat(alpha, n_clients), size=len(labels))
        client_indices: List[List[int]] = [[] for _ in range(n_clients)]

        for j, lab in enumerate(labels):
            pool = class_idx[lab]
            if len(pool) == 0:
                continue
            share = np.maximum(proportions[j] / proportions[j].sum() * len(pool), floor)
            counts = _allocate(share, len(pool))
            for client, chunk in enumerate(np.split(pool, np.cumsum(counts)[:-1])):
                client_indices[client].extend(int(v) for v in chunk)

        sizes = [len(c) for c in client_indices]
        if min(sizes) >= min_samples:
            return [
                ds.subset(np.sort(np.asarray(sorted(c), dtype=int)))
                for c in client_indices
            ]

    raise RuntimeError(
        f"could not build a Dirichlet partition with >= {min_samples} samples per "
        f"client (alpha={alpha}, n_clients={n_clients}). Try a larger alpha or "
        "fewer clients."
    )


def partition(
    ds: Dataset,
    n_clients: int,
    mode: str = "iid",
    alpha: float = 0.5,
    seed: int = 0,
    min_samples: int = 5,
) -> List[Dataset]:
    if mode == "iid":
        return iid_partition(ds, n_clients, seed=seed)
    if mode == "dirichlet":
        return dirichlet_partition(
            ds, n_clients, alpha=alpha, seed=seed, min_samples=min_samples
        )
    raise ValueError(f"unknown partition mode: {mode!r}")


def summarize(clients: Sequence[Dataset]) -> List[dict]:
    """Per-client sample counts and positive-label rate -- printed at startup."""
    rows = []
    for i, c in enumerate(clients):
        rows.append(
            {
                "client": f"client-{i:02d}",
                "samples": len(c),
                "positive_rate": round(float(np.mean(c.y)), 3) if len(c) else 0.0,
            }
        )
    return rows
