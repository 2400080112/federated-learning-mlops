# Federated Learning with MLOps

A PyCharm-ready educational demo of federated learning with a small MLOps workflow. It trains a logistic-regression model over simulated clients using the Wisconsin Diagnostic Breast Cancer dataset bundled with the project.

## What it demonstrates

- Deterministic partitioning of data across clients
- Client-side training; only model parameters and sample counts are sent to the aggregator
- Federated averaging (FedAvg) over multiple rounds
- Central evaluation after each round
- Versioned model artifacts, metrics, and a `registry/latest.json` pointer
- A local prediction command that loads the latest registered model

This runs all clients in one Python process to make the workflow easy to learn. It is not a production deployment and does not provide formal privacy guarantees such as differential privacy or secure aggregation. The example dataset is public and is used for demonstration only.

## Run in PyCharm

1. Open the `federated-mlops` folder in PyCharm.
2. Create a Python 3.10+ virtual environment when prompted.
3. In PyCharm's terminal, install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run `main.py` from the project root, or use:

   ```bash
   python main.py train --clients 5 --rounds 8
   python main.py predict --features 14.2 20.1 94.5 600 0.1 0.2 0.15 0.08 0.18 0.06 0.4 1.2 2.5 35 0.01 0.05 0.04 0.02 0.02 0.003 16 25 105 800 0.14 0.5 0.5 0.2 0.3 0.08
   ```

The features must be 30 raw numeric values in the dataset's feature order. The saved preprocessing parameters are applied by the prediction command. This example predicts from the latest registered model.

## Project layout

```text
federated-mlops/
  config.json
  main.py
  requirements.txt
  src/fedmlops/
    __init__.py
    data.py
    model.py
    pipeline.py
    registry.py
```

Generated artifacts are written under `artifacts/` and `registry/` after training. Each training run creates a timestamped version directory with weights, configuration, and metrics.

## Workflow

1. Load and split the dataset into a training set and a held-out test set.
2. Split training indices among simulated clients.
3. For each round, each client trains from the current global model on its own partition.
4. The server combines the client weights using sample-count-weighted FedAvg.
5. Evaluate the global model centrally and record round metrics.
6. Save the final version and update the local model registry pointer.

## Privacy and production extensions

Client records remain local to each client function during training, but this simulation runs on one machine and model updates can still leak information. For real privacy-preserving deployment, add secure aggregation, differential privacy, authenticated transport, client orchestration, monitoring, and a governed model registry.
