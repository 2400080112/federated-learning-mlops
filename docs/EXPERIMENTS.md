# Experiments: the federated learning method

How the model is trained, which knobs exist, what each one does to the numbers,
and how to reproduce every row of the results table.

## The algorithm

One round of FedAvg as implemented in `backend/fedmlops/trainer.py`:

```
for round t = 1..T:
    S ← random subset of clients of size clients_per_round
    for each client i in S:                    # data stays here
        θ_i ← θ_global
        for e = 1..E:                          # local epochs
            for each mini-batch of client i's rows:
                θ_i ← θ_i − η · ∇L_i(θ_i)      # BCE + L2, plain SGD
        Δ_i ← θ_i − θ_global                   # only this leaves the client
        if DP:  Δ_i ← clip(Δ_i, C)             # bound sensitivity
    w_i ← n_i / Σn                             # sample-count weights
    Δ   ← Σ_i w_i · Δ_i
    if DP:  Δ ← Δ + N(0, (σ · max_i(w_i) · C)²)   # noise on the aggregate
    θ_global ← θ_global + Δ
    evaluate on the validation split; keep the best checkpoint
```

Two details that are easy to get wrong and are worth pointing at:

- **The noise is calibrated to `max_i(w_i)·C`, not `C`.** The sensitivity of a
  weighted *mean* of clipped deltas is the largest weight times the clipping
  bound. Using `C` would over-noise the aggregate.
- **The registered model is the best validation checkpoint**, not the last round.
  With DP on, the last round is often noisier than round 12.

## Hyperparameters

| Parameter | Default | Where | Effect |
|---|---|---|---|
| `n_clients` | 8 | Train form / API | more clients = smaller, noisier local datasets |
| `clients_per_round` | = `n_clients` | Train form / API | partial participation simulates dropout; hurts convergence |
| `rounds` | 25 | Train form / API | more rounds ≈ better fit, more ε spent |
| `local_epochs` (E) | 3 | Train form / API | more local work per round = fewer rounds needed, more client drift |
| `batch_size` | 16 | Train form / API | SGD noise |
| `learning_rate` (η) | 0.15 | Train form / API | 0.15 works well for standardised features |
| `l2` | 0.001 | Train form / API | mild regularisation |
| `partition` | `iid` | Train form / API | `dirichlet` for label skew |
| `dirichlet_alpha` | 0.5 | Train form / API | 0.1 = severe skew, 1.0 = mild |
| `seed` | 7 | Train form / API | controls partition, client sampling and DP noise |
| `dp_epsilon` | off | Train form / API | σ is derived from ε and the round count |
| `dp_clip` (C) | 1.0 | Train form / API | update clipping bound |

Split sizes are fixed by the config: test 20%, validation 20%, rest for clients.

## Reproducing the results

Every row of the model card's metrics table comes from one of these. Equivalent
via the UI: Train page with the same settings, then *Score* and *Register* in the
run drawer.

```bash
# centralised reference: all rows pooled
python run.py baseline            # CLI build (federated-mlops-demo)

# federated, IID, 8 clients, 25 rounds
curl -X POST localhost:8000/api/runs -H 'content-type: application/json' \
  -d '{"name":"iid","n_clients":8,"rounds":25,"seed":7}'

# non-IID: Dirichlet α=0.3, 10 clients, 30 rounds
curl -X POST localhost:8000/api/runs -H 'content-type: application/json' \
  -d '{"name":"noniid","n_clients":10,"rounds":30,"partition":"dirichlet","dirichlet_alpha":0.3,"seed":11}'

# differential privacy, ε = 8
curl -X POST localhost:8000/api/runs -H 'content-type: application/json' \
  -d '{"name":"dp8","n_clients":10,"rounds":20,"privacy_enabled":true,"dp_epsilon":8.0,"seed":5}'
```

Then, for each run id: `POST /api/runs/{id}/evaluate` and read
`test_metrics` from `GET /api/runs/{id}`.

`python -m app.seed --reset` does all three in one go and prints the numbers.

## What the experiments show

**1. Federated ≈ centralised on easy data.** 0.9823 vs 0.9911 test accuracy — one
row of 113. WDBC is nearly linearly separable, so averaging local models loses
almost nothing. This is a property of the dataset, not a general result.

**2. Label skew costs more than federation.** Dirichlet α=0.3 (clients holding
mostly one class) drops test accuracy to 0.9735 and F1 to 0.9790 while validation
accuracy still reads 1.00 — a reminder that a client with 4 malignant rows can
score its own data perfectly while contributing a biased update.

**3. Privacy has a visible price.** ε≈8 → 0.9204, ε≈1 → ~0.90. Two caveats worth
stating whenever these numbers are quoted: the ε is a loose Gaussian-composition
bound (no RDP, no subsampling amplification), so the *real* cost of ε=1 is much
smaller than this table suggests; and clipping fraction (logged per round, visible
in the run drawer) tells you how much of the loss is clipping rather than noise.

**4. Client participation matters.** With `clients_per_round < n_clients`, each
round sees a different subset, so the loss curve is noisier and needs more rounds
for the same accuracy. Worth demonstrating live — it is the single easiest knob
to feel.

## Things this study does not measure

- **Multiple seeds.** Every number above is one seed. Differences of a row or two
  are noise; run 5 seeds before treating a delta as real.
- **Hard data.** No CIFAR/MNIST-scale experiment, where non-IID federated
  training genuinely struggles.
- **Communication cost.** Rounds are the unit here, not bytes. A real study would
  report bytes uploaded per round and the accuracy-per-byte frontier.
- **Privacy attacks.** No membership-inference attack is run, so "privacy-
  preserving" is an argument about the mechanism, not a measurement of leakage.
  [PRIVACY.md](PRIVACY.md) says what would need to be built.
