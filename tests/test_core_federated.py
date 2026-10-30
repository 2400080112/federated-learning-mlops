"""The federated core, exercised directly (no HTTP, no database).

The web layer is only worth anything if the maths under it is right, so these
tests check the properties the API relies on: FedAvg is a weighted mean, the
partition covers every row, clipping bounds the norm, and the trainer improves.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedmlops.config import RunConfig
from fedmlops.data import Dataset, load_splits
from fedmlops.metrics import evaluate, roc_auc
from fedmlops.model import LogisticRegression, fedavg
from fedmlops.partition import dirichlet_partition, iid_partition
from fedmlops.privacy import PrivacyAccountant, clip_update, sigma_for_epsilon
from fedmlops.trainer import train_federated


@pytest.fixture(scope="module")
def splits():
    return load_splits(
        "data/breast_cancer.csv",
        target_column="target",
        drop_columns=("sample_id", "target_name"),
        test_size=0.2,
        val_size=0.2,
        seed=7,
    )


@pytest.fixture
def toy() -> Dataset:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-1, 0.5, (40, 2)), rng.normal(1, 0.5, (40, 2))])
    y = np.array([0] * 40 + [1] * 40)
    return Dataset(X=X, y=y, feature_names=["f0", "f1"], mean=np.zeros(2), std=np.ones(2))


def test_fedavg_is_a_size_weighted_mean():
    assert np.allclose(fedavg([np.array([1.0, 1.0]), np.array([3.0, 3.0])], [1, 3]), [2.5, 2.5])
    with pytest.raises(ValueError):
        fedavg([], [])


def test_partitions_cover_every_row(splits):
    train, _, _ = splits
    for clients in (iid_partition(train, 6, seed=1), dirichlet_partition(train, 6, alpha=0.3, seed=1)):
        assert sum(len(c) for c in clients) == len(train)
        assert all(len(c) > 0 for c in clients)


def test_model_learns_and_serialises(toy):
    model = LogisticRegression(n_features=2)
    before = model.loss(toy.X, toy.y)
    model.local_train(toy.X, toy.y, epochs=60, learning_rate=0.5, batch_size=32,
                      rng=np.random.default_rng(0))
    assert model.loss(toy.X, toy.y) < before
    clone = LogisticRegression.from_dict(model.to_dict())
    assert np.array_equal(clone.get_params(), model.get_params())


def test_metrics_are_sane():
    assert roc_auc(np.array([0, 1]), np.array([0.1, 0.9])) == pytest.approx(1.0)
    scores = evaluate(np.array([0, 1, 1, 0]), np.array([0.2, 0.8, 0.7, 0.4]))
    assert scores["accuracy"] == 1.0


def test_clipping_bounds_the_update():
    clipped, raw = clip_update(np.array([3.0, 4.0]), clip_norm=1.0)
    assert raw == pytest.approx(5.0)
    assert np.linalg.norm(clipped) == pytest.approx(1.0)


def test_epsilon_and_sigma_round_trip():
    sigma = sigma_for_epsilon(epsilon=2.0, rounds=25, delta=1e-5)
    accountant = PrivacyAccountant(sigma=sigma, delta=1e-5)
    for _ in range(25):
        accountant.spend_round()
    assert accountant.epsilon == pytest.approx(2.0, rel=1e-9)


def test_trainer_reports_every_round_and_improves(splits):
    train, val, _ = splits
    cfg = RunConfig()
    cfg.federated.n_clients = 5
    cfg.federated.clients_per_round = 5
    cfg.federated.rounds = 8
    cfg.federated.local_epochs = 3

    seen = []
    result = train_federated(train, val, cfg, on_round=seen.append)

    assert len(seen) == 8 == len(result.history)
    assert [r.round for r in seen] == list(range(1, 9))
    assert result.history[-1].val_accuracy > result.history[0].val_accuracy
    assert result.val_metrics["accuracy"] > 0.85
    assert result.best_round >= 1
