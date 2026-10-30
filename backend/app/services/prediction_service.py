"""Running inference from a registered model, and recording what was predicted.

The scaler travels with the model (``scaler_json``), so a request never has to
know how the training data was standardised -- a small thing that is easy to get
wrong and silently ruins predictions when you do.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from fedmlops.data import resolve_path
from fedmlops.model import LogisticRegression

from ..db import PROJECT_ROOT
from ..models import ModelVersion, Prediction
from .registry_service import ServiceError

PREDICTION_DIR = PROJECT_ROOT / "data" / "predictions"
BUNDLED_DATASET = PROJECT_ROOT / "data" / "breast_cancer.csv"
PREVIEW_ROWS = 25


def _load_frame(
    csv_bytes: Optional[bytes], use_bundled: bool, limit: Optional[int]
) -> Tuple[pd.DataFrame, str]:
    if csv_bytes:
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes))
        except Exception as exc:  # noqa: BLE001
            raise ServiceError(f"could not parse the uploaded CSV: {exc}") from exc
        source = "uploaded CSV"
    elif use_bundled:
        path = resolve_path(BUNDLED_DATASET)
        if not path.exists():
            raise ServiceError(f"bundled dataset missing at {path}")
        df = pd.read_csv(path)
        source = "bundled breast_cancer.csv"
    else:
        raise ServiceError("nothing to predict: upload a CSV or set use_bundled_dataset")

    if df.empty:
        raise ServiceError("the input has no rows")
    if limit:
        df = df.head(limit)
    return df, source


def predict(
    session: Session,
    model_version: ModelVersion,
    csv_bytes: Optional[bytes] = None,
    use_bundled: bool = True,
    threshold: float = 0.5,
    limit: Optional[int] = None,
) -> Prediction:
    scaler = json.loads(model_version.scaler_json or "{}")
    feature_names: List[str] = scaler.get("feature_names") or []
    if not feature_names:
        raise ServiceError(
            f"{model_version.name}:{model_version.version} has no scaler metadata; "
            "re-register the run"
        )

    df, source = _load_frame(csv_bytes, use_bundled, limit)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ServiceError(
            f"input is missing {len(missing)} feature column(s): {missing[:5]}"
        )

    raw = df[feature_names].to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ServiceError("input contains NaN or infinite values in feature columns")

    mean = np.asarray(scaler["mean"], dtype=float)
    std = np.asarray(scaler["std"], dtype=float)
    std = np.where(std == 0, 1.0, std)
    X = (raw - mean) / std

    model = LogisticRegression.from_dict(json.loads(model_version.model_json))
    probs = model.predict_proba(X)
    preds = (probs >= threshold).astype(int)

    out = df.copy()
    out["p_benign"] = np.round(probs, 6)
    out["prediction"] = preds
    out["prediction_label"] = np.where(preds == 1, "benign", "malignant")

    summary: Dict[str, Any] = {
        "n_rows": int(len(out)),
        "predicted_benign": int(preds.sum()),
        "predicted_malignant": int((preds == 0).sum()),
        "mean_confidence": round(float(np.mean(np.abs(probs - 0.5) * 2)), 4),
        "threshold": threshold,
    }

    if "target" in out.columns:
        truth = out["target"].to_numpy(dtype=int)
        tp = int(np.sum((preds == 1) & (truth == 1)))
        tn = int(np.sum((preds == 0) & (truth == 0)))
        fp = int(np.sum((preds == 1) & (truth == 0)))
        fn = int(np.sum((preds == 0) & (truth == 1)))
        summary.update(
            {
                "accuracy": round(float(np.mean(preds == truth)), 4),
                "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                "evaluated_against": "target column present in input",
            }
        )

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = PREDICTION_DIR / f"predictions-{stamp}.csv"
    out.to_csv(out_path, index=False)

    preview_cols = [c for c in ("sample_id", "target", "prediction_label", "p_benign") if c in out.columns]
    preview = out[preview_cols].head(PREVIEW_ROWS).to_dict(orient="records")

    record = Prediction(
        model_version_id=model_version.id,
        model_label=f"{model_version.name}:{model_version.version}",
        source_label=source,
        threshold=threshold,
        n_rows=int(len(out)),
        summary_json=json.dumps(summary),
        preview_json=json.dumps(preview, default=str),
        output_path=str(out_path),
    )
    session.add(record)
    session.commit()
    return record


def to_out(record: Prediction) -> Dict[str, Any]:
    return {
        "id": record.id,
        "created_at": record.created_at,
        "model_label": record.model_label,
        "source_label": record.source_label,
        "threshold": record.threshold,
        "n_rows": record.n_rows,
        "summary": json.loads(record.summary_json or "{}"),
        "preview": json.loads(record.preview_json or "[]"),
        "download_url": f"/api/predictions/{record.id}/download",
    }


def output_file(record: Prediction) -> Path:
    if not record.output_path:
        raise ServiceError("this prediction has no stored CSV")
    path = Path(record.output_path)
    if not path.exists():
        raise ServiceError("the prediction CSV is no longer on disk")
    return path
