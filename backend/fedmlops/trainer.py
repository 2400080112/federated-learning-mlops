"""The federated training loop (and a centralised baseline to compare against).

One round is: pick a subset of clients -> send the global parameters -> each
client runs a few epochs of local SGD -> clip/noise the resulting delta if DP is
on -> size-weighted average -> validate on the held-out validation split. Only
parameters ever leave a client; the rows never do. That is the whole point of
the demo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import metrics as M
from .config import RunConfig
from .data import Dataset
from .model import LogisticRegression, fedavg
from .partition import partition, summarize
from .privacy import PrivacyAccountant, add_gaussian_noise, clip_update

Logger = Optional[Callable[[str], None]]


@dataclass
class RoundRecord:
    round: int
    clients: int
    samples: int
    train_loss: float
    val_accuracy: float
    val_f1: float
    val_roc_auc: float
    val_log_loss: float
    mean_update_norm: float
    clipped_fraction: float
    epsilon: float


@dataclass
class TrainResult:
    model: LogisticRegression
    history: List[RoundRecord] = field(default_factory=list)
    clients: List[dict] = field(default_factory=list)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    best_round: int = 0
    privacy: Dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0
    mode: str = "federated"

    def history_as_dicts(self) -> List[dict]:
        return [vars(r) for r in self.history]


def _noop(_: str) -> None:
    return None


def train_federated(
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: RunConfig,
    log: Logger = None,
    rng: Optional[np.random.Generator] = None,
    on_round: Optional[Callable[["RoundRecord"], None]] = None,
) -> TrainResult:
    """Run the federated loop.

    ``on_round`` is called after every round with that round's record. The web
    backend uses it to stream progress into the database while training runs;
    a caller that does not care can ignore it and read ``result.history`` at the
    end instead.
    """
    log = log or _noop
    fed, priv = cfg.federated, cfg.privacy
    rng = rng or np.random.default_rng(cfg.seed)
    started = time.time()

    clients = partition(
        train_ds,
        n_clients=fed.n_clients,
        mode=fed.partition,
        alpha=fed.dirichlet_alpha,
        seed=cfg.seed,
        min_samples=fed.min_client_samples,
    )
    client_summary = summarize(clients)
    log(
        f"{len(clients)} clients | partition={fed.partition}"
        + (f" (alpha={fed.dirichlet_alpha})" if fed.partition == "dirichlet" else "")
        + f" | {len(train_ds)} training rows"
    )
    for row in client_summary:
        log(
            f"  {row['client']}: {row['samples']:>3} samples, "
            f"positive_rate={row['positive_rate']:.2f}"
        )

    global_model = LogisticRegression(n_features=train_ds.X.shape[1], l2=fed.l2)
    accountant = PrivacyAccountant(
        sigma=priv.noise_multiplier,
        delta=priv.delta,
        max_epsilon=priv.max_epsilon,
    )

    history: List[RoundRecord] = []
    best_params = global_model.get_params()
    best_val: Dict[str, float] = {}
    best_round = 0

    for round_no in range(1, fed.rounds + 1):
        active = rng.choice(
            len(clients), size=fed.clients_per_round, replace=False
        )
        active = np.sort(active)

        global_params = global_model.get_params()
        deltas: List[np.ndarray] = []
        sizes: List[int] = []
        losses: List[float] = []
        norms: List[float] = []
        clipped = 0

        for cid in active:
            client_ds = clients[cid]
            local = LogisticRegression(
                n_features=global_model.n_features, l2=fed.l2
            ).set_params(global_params)
            local.local_train(
                client_ds.X,
                client_ds.y,
                epochs=fed.local_epochs,
                learning_rate=fed.learning_rate,
                batch_size=fed.batch_size,
                rng=rng,
            )
            delta = local.get_params() - global_params
            losses.append(local.loss(client_ds.X, client_ds.y))

            if priv.enabled:
                delta, raw_norm = clip_update(delta, priv.clip_norm)
                if raw_norm > priv.clip_norm:
                    clipped += 1
            else:
                raw_norm = float(np.linalg.norm(delta))
            norms.append(raw_norm)
            deltas.append(delta)
            sizes.append(len(client_ds))

        weights = np.asarray(sizes, dtype=float) / float(sum(sizes))
        stacked = np.stack(deltas, axis=0)
        aggregate = weights @ stacked

        if priv.enabled:
            # sensitivity of a weighted mean of clipped deltas is max_w * C, so
            # that -- not C -- is what the noise has to scale with.
            sensitivity = float(np.max(weights)) * priv.clip_norm
            aggregate = aggregate + rng.normal(
                loc=0.0,
                scale=priv.noise_multiplier * sensitivity,
                size=aggregate.shape,
            )

        global_model.set_params(global_params + aggregate)
        epsilon = accountant.spend_round() if priv.enabled else 0.0

        val_prob = global_model.predict_proba(val_ds.X)
        val = M.evaluate(val_ds.y, val_prob)
        record = RoundRecord(
            round=round_no,
            clients=int(len(active)),
            samples=int(sum(sizes)),
            train_loss=round(float(np.mean(losses)), 6),
            val_accuracy=val["accuracy"],
            val_f1=val["f1"],
            val_roc_auc=val["roc_auc"],
            val_log_loss=val["log_loss"],
            mean_update_norm=round(float(np.mean(norms)), 6),
            clipped_fraction=round(clipped / max(1, len(active)), 4),
            epsilon=round(float(epsilon), 4),
        )
        history.append(record)
        if on_round is not None:
            on_round(record)
        log(
            f"round {round_no:>2}/{fed.rounds} | clients={record.clients} | "
            f"loss={record.train_loss:.4f} | val_acc={record.val_accuracy:.4f} | "
            f"val_auc={record.val_roc_auc:.4f} | |delta|={record.mean_update_norm:.3f}"
            + (f" | eps={record.epsilon:.2f}" if priv.enabled else "")
        )

        if val["accuracy"] > best_val.get("accuracy", -1.0):
            best_val, best_params, best_round = val, global_model.get_params(), round_no

        if accountant.budget_exceeded:
            log(
                f"privacy budget exhausted at round {round_no} "
                f"(eps={accountant.epsilon:.2f} > {priv.max_epsilon}); stopping"
            )
            break

    final_model = LogisticRegression(
        n_features=global_model.n_features, l2=fed.l2
    ).set_params(best_params)

    privacy_report: Dict[str, float] = {"enabled": bool(priv.enabled)}
    if priv.enabled:
        privacy_report.update(
            {
                "clip_norm": priv.clip_norm,
                "noise_multiplier": priv.noise_multiplier,
                "delta": priv.delta,
                "epsilon": round(accountant.epsilon, 4),
                "rounds": accountant.rounds_done,
                "accounting": "basic Gaussian composition (sqrt(2k ln(1/delta))/sigma)",
            }
        )

    return TrainResult(
        model=final_model,
        history=history,
        clients=client_summary,
        val_metrics=best_val,
        best_round=best_round,
        privacy=privacy_report,
        duration_s=round(time.time() - started, 3),
        mode="federated",
    )


def train_centralized(
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: RunConfig,
    epochs: int = 40,
    log: Logger = None,
) -> TrainResult:
    """Pooled-data baseline: everything in one place, same model and loss.

    This is the reference point the federated run is measured against -- the
    gap between the two is the price of keeping data local. Not a tuned
    baseline; just a fair one.
    """
    log = log or _noop
    rng = np.random.default_rng(cfg.seed)
    started = time.time()
    fed = cfg.federated

    model = LogisticRegression(n_features=train_ds.X.shape[1], l2=fed.l2)
    history: List[RoundRecord] = []
    best_params, best_val, best_round = model.get_params(), {}, 0

    for epoch in range(1, epochs + 1):
        model.local_train(
            train_ds.X,
            train_ds.y,
            epochs=1,
            learning_rate=fed.learning_rate,
            batch_size=fed.batch_size,
            rng=rng,
        )
        val = M.evaluate(val_ds.y, model.predict_proba(val_ds.X))
        history.append(
            RoundRecord(
                round=epoch,
                clients=1,
                samples=len(train_ds),
                train_loss=round(model.loss(train_ds.X, train_ds.y), 6),
                val_accuracy=val["accuracy"],
                val_f1=val["f1"],
                val_roc_auc=val["roc_auc"],
                val_log_loss=val["log_loss"],
                mean_update_norm=0.0,
                clipped_fraction=0.0,
                epsilon=0.0,
            )
        )
        if val["accuracy"] > best_val.get("accuracy", -1.0):
            best_val, best_params, best_round = val, model.get_params(), epoch

    log(f"centralised baseline: best val_acc={best_val.get('accuracy', float('nan')):.4f}")
    return TrainResult(
        model=LogisticRegression(
            n_features=model.n_features, l2=fed.l2
        ).set_params(best_params),
        history=history,
        clients=[{"client": "pooled", "samples": len(train_ds),
                  "positive_rate": round(float(np.mean(train_ds.y)), 3)}],
        val_metrics=best_val,
        best_round=best_round,
        privacy={"enabled": False},
        duration_s=round(time.time() - started, 3),
        mode="centralized",
    )
