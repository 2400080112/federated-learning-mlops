# Privacy: what this app implements, and what it does not

Short version: there is a real Gaussian mechanism over clipped client updates,
plus a deliberately loose epsilon bound. That is enough to *study* the
accuracy-versus-privacy trade-off. It is not enough to make a privacy claim
about real patient data.

## The mechanism

For each round and each participating client `i` (`backend/fedmlops/privacy.py`,
called from `trainer.py`):

1. **Clip.** `delta_i = theta_local - theta_global`, scaled so
   `||delta_i||_2 <= C` (`privacy.clip_norm`). Clipping bounds the *sensitivity*
   — how much one client's data can move the aggregate — which is what makes the
   noise scale meaningful. Without clipping, "adding noise" is decoration.
2. **Average with weights.** `w_i = n_i / sum(n)`. The sensitivity of that
   weighted mean is `max_i(w_i) * C`, so that — not `C` — is what the noise is
   calibrated to.
3. **Add noise.** `aggregate += N(0, (sigma * max_i(w_i) * C)^2)` per coordinate.

Reported epsilon uses basic Gaussian-mechanism composition over `k` rounds:

```
epsilon ~= sqrt(2 * k * ln(1/delta)) / sigma
```

`sigma_for_epsilon()` inverts it, which is how the Train form's ε field picks a
noise level for you: fix ε, and the app derives σ from your round count.

## Why the number is loose

The bound ignores two things a real accountant uses, both of which make the true
epsilon much smaller:

- **Rényi / moments accounting (RDP)** instead of linear composition.
- **Subsampling amplification.** When clients-per-round < clients, each round
  only touches a random subset, which amplifies privacy substantially. This app
  composes as if every client took part every round — the pessimistic case.

Practical consequence: ε = 1 for 20 rounds produces σ ≈ 21, far more noise than
a properly accounted ε = 1 needs, and accuracy drops to ~0.92 on a task that is
otherwise ~0.98. Read that as "the cost of ε = 1 under a crude bound", not as
"the cost of ε = 1".

## Leakage this app does not fix

- **No secure aggregation.** The server sees each client's individual update.
  Here the "clients" are objects in one Python process, so there is no transport
  at all — the point is that the *design* keeps rows local, not that the channel
  is protected. DP is the mitigation implemented.
- **The shared scaler.** Feature mean/std are computed on pooled training data
  (`fedmlops/data.py:standardize`). Aggregates, not rows, but still computed
  centrally.
- **No membership-inference test.** There is no attack script in this repo. If
  you want to argue privacy empirically, that is the experiment to add: measure
  attack advantage with and without DP.
- **No auth on the API.** Anyone who can reach the port can train, register,
  promote and predict. Localhost only.

## Upgrade path, in order of value

1. Replace the composition bound with an RDP accountant and re-measure — the
   accuracy–epsilon curve gets much friendlier.
2. Add subsampling amplification to the accountant.
3. Add secure aggregation (even a toy pairwise-masking scheme over three
   clients) so the server cannot read individual updates.
4. Add a membership-inference attack script and report attack advantage next to
   epsilon. That is what turns "privacy-preserving" from an adjective into a
   measurement.
