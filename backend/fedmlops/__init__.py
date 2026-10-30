"""Core federated-learning library (no web, no database).

Pure NumPy: data loading, client partitioning, logistic regression, FedAvg,
differential privacy. The FastAPI layer in ``backend/app`` imports from here and
adds persistence, orchestration and the HTTP API on top.

Keeping this layer free of FastAPI/SQLAlchemy is deliberate -- the training
maths stays testable on its own, and the same code could be driven by a CLI, a
notebook or a test without a server.
"""

__version__ = "0.2.0"
