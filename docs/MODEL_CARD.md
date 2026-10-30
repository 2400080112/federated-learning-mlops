# Model card — federated WDBC classifier

What this model is, what it was trained on, how well it works, and where it must
not be used. Written in the spirit of Mitchell et al., *Model Cards for Model
Reporting* (2019).

## Model details

| | |
|---|---|
| **Task** | Binary classification: benign vs malignant breast mass |
| **Model** | Logistic regression, 30 features + bias (31 parameters) |
| **Loss** | Binary cross-entropy with L2 penalty on the weights |
| **Optimiser** | Mini-batch SGD, run locally on each client |
| **Aggregation** | FedAvg — sample-count-weighted mean of client parameter deltas |
| **Privacy** | Optional: per-client update clipping to L2 norm `C`, Gaussian noise on the aggregate, ε/δ reported |
| **Implementation** | `backend/fedmlops/model.py`, `trainer.py`, `privacy.py` — plain NumPy, no ML framework |
| **Output** | `p_benign ∈ [0,1]`; thresholded at 0.5 by default (adjustable at serving time) |

Deliberately a linear model: the point of the project is the federated training
loop and the operational workflow around it, and a model you can read end to end
keeps the maths visible. Nothing is hidden in a framework.

## Training data

**Wisconsin Diagnostic Breast Cancer (WDBC)** — UCI ML Repository (Street,
Wolberg & Mangasarian, 1993), as distributed with scikit-learn. Bundled at
`data/breast_cancer.csv`.

- 569 samples, 30 real-valued features computed from digitised fine-needle
  aspirate images: mean / standard-error / worst of radius, texture, perimeter,
  area, smoothness, compactness, concavity, concave points, symmetry, fractal
  dimension.
- Labels: **1 = benign (357)**, **0 = malignant (212)**. Mildly imbalanced
  (63/37).
- Split, once, stratified, seed 7: **342 train / 114 validation / 113 test**.
  The validation split drives per-round monitoring and checkpoint selection; the
  test split is touched once, for the reported number.
- Standardisation: mean/std fitted on the **pooled training split** only. This is
  the one place the pipeline touches all rows at once — see "Limitations".

## Federated setup

The training split is partitioned across simulated clients; no client row is
ever shared with the server, only parameter deltas.

| Regime | How clients are formed | Effect |
|---|---|---|
| `iid` | shuffle and cut into equal chunks | every client has a similar class balance; FedAvg ≈ centralised |
| `dirichlet` | each class dealt out in `Dirichlet(α)` proportions, with a floor | low α produces clients that hold almost one class — the realistic, hard case |

## Metrics

Test split (113 rows), single seed, produced by the commands in
[EXPERIMENTS.md](EXPERIMENTS.md):

| Configuration | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Centralised reference (all data pooled) | 0.9911 | — | — | 0.9929 | — |
| Federated, 8 clients, IID, 25 rounds | 0.9823 | 0.9859 | 0.9859 | 0.9859 | 0.9993 |
| Federated, 10 clients, Dirichlet α=0.3, 30 rounds | 0.9735 | — | — | 0.9790 | — |
| Federated, 10 clients, DP ε≈8 | 0.9204 | — | — | 0.9390 | — |

Validation accuracy across configurations sits at 0.97–1.00. The number that
matters for the project's argument is the last row: **privacy costs measurable
accuracy**, and the app lets you move ε and watch that trade directly.

## Intended use

- **Teaching and coursework**: a working, inspectable federated learning loop
  with an MLOps workflow around it.
- **Method study**: comparing IID vs non-IID partitioning, full vs partial client
  participation, and the accuracy cost of differential privacy.
- **Systems demonstration**: runs, registry versions, promotion gates, serving.

## Out-of-scope use

- **Not a diagnostic device.** It is a demo on 569 historical samples with no
  clinical validation, no calibration study, no external cohort, and no
  regulatory review. Do not use it to make, support, or discuss a decision about
  a person's health.
- **Not a privacy guarantee.** The ε reported is a loose Gaussian-composition
  bound, not RDP with subsampling amplification, and there is no secure
  aggregation — see [PRIVACY.md](PRIVACY.md). The demo is not safe for real
  patient data.
- **Not a benchmark.** One dataset, one seed, one linear model. The
  federated-vs-centralised gap here (one test row) says nothing about federated
  learning on hard, high-dimensional, genuinely non-IID data.

## Limitations and biases

- **Small, easy, nearly separable data.** WDBC is 569 rows with a strongly
  linear decision boundary. Non-IID skew barely hurts here; on CIFAR-10 or real
  hospital data it hurts a lot. Treat the non-IID row above as a floor on the
  damage, not a typical estimate.
- **Single seed.** Numbers are not averaged over seeds, so differences of one or
  two test rows are noise. Run `--seed` variants before claiming a difference.
- **Centrally fitted scaler.** Feature standardisation uses pooled training
  statistics. In a strict federated setting that is a (small) leak; a real
  deployment standardises locally or shares noisy statistics.
- **No subgroup analysis.** No breakdown by patient age, site, scanner or
  acquisition protocol — WDBC does not carry those fields. A model that is 98%
  accurate overall can still be much worse on a subgroup, and this card cannot
  tell you.
- **Label provenance.** Labels are from the original 1990s dataset; they reflect
  the diagnostic practice and population of that time and place.
- **Threshold is a choice.** At 0.5 the model trades recall against precision
  invisibly. For a screening-like use you would lower the threshold and report
  the recall you actually achieved — the Predict page exposes the slider so that
  trade is visible rather than buried.

## Ethical considerations

Medical classification demos are easy to over-read. The honest framing is: this
project demonstrates *how* a model could be trained without pooling patient data,
and what operational discipline (measured before registered, gated promotion,
versioned artifacts) that requires. It does not demonstrate a clinically useful
model, and the difference matters.

## Files

- Weights + scaler: `model_versions.model_json` / `scaler_json` in the database,
  and per-run `model_json` on the run row.
- Training code: `backend/fedmlops/`
- Reproduction: [EXPERIMENTS.md](EXPERIMENTS.md)
