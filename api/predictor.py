"""
FraudPredictor: loads the trained model + preprocessor and exposes
single / batch prediction used by the FastAPI layer.

The preprocessing step must reproduce *exactly* the transformations
applied at training time (scaling of Amount/Time + temporal feature
engineering) and the model must receive features in the same order
it was fitted on.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .schemas import PredictionOutput

logger = logging.getLogger(__name__)

# Canonical feature order the model was fitted on. This mirrors the
# output of fraudDetectionPreprocessor (V1-V28, Time, Amount, then the
# engineered temporal features).
EXPECTED_FEATURES: List[str] = (
    [f"V{i}" for i in range(1, 29)]
    + ["Time", "Amount", "HourOfDay", "IsNight", "IsWeekend", "LogAmount"]
)

DEFAULT_THRESHOLD = 0.5
METRICS_PATH = Path("reports/metrics.json")


def _risk_level(probability: float) -> str:
    if probability >= 0.80:
        return "Critical"
    if probability >= 0.60:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


class FraudPredictor:
    """Wrapper around the trained model + preprocessor."""

    def __init__(
        self,
        model_path: str = "models/fraud_detector_model.joblib",
        preprocessor_path: str = "models/preprocessor.joblib",
        metrics_path: str = "reports/metrics.json",
        threshold: Optional[float] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.metrics_path = Path(metrics_path)

        self.model: Any = None
        self.preprocessor: Any = None
        self.threshold: float = DEFAULT_THRESHOLD
        self.metrics: Dict[str, Any] = {}
        self.feature_names_: List[str] = []
        self.is_loaded: bool = False

        # Resolve threshold: explicit arg > metrics file > default
        if threshold is not None:
            self.threshold = float(threshold)
        else:
            self.threshold = self._load_threshold_from_metrics()

        self.load()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load_threshold_from_metrics(self) -> float:
        try:
            if Path(self.metrics_path).exists():
                with open(self.metrics_path, "r") as f:
                    metrics = json.load(f)
                opt = metrics.get("optimal_threshold")
                if opt is not None:
                    logger.info("Using optimal threshold from metrics: %.4f", float(opt))
                    return float(opt)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read metrics file: %s", exc)
        return DEFAULT_THRESHOLD

    def load(self) -> "FraudPredictor":
        """Load model and preprocessor from disk."""
        logger.info("Loading model from %s", self.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)

        logger.info("Loading preprocessor from %s", self.preprocessor_path)
        if self.preprocessor_path.exists():
            self.preprocessor = joblib.load(self.preprocessor_path)
        else:
            logger.warning("Preprocessor file not found; raw inputs will be used")
            self.preprocessor = None

        # Capture the feature order the model was fitted on, if available.
        if hasattr(self.model, "feature_names_in_"):
            self.feature_names_ = list(self.model.feature_names_in_)
        elif EXPECTED_FEATURES:
            self.feature_names_ = list(EXPECTED_FEATURES)

        self.is_loaded = True
        logger.info(
            "FraudPredictor ready (features=%d, threshold=%.3f)",
            len(self.feature_names_),
            self.threshold,
        )
        return self

    # ------------------------------------------------------------------ #
    # Preprocessing helpers
    # ------------------------------------------------------------------ #
    def _raw_frame(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """Build a raw input DataFrame with the columns the preprocessor expects."""
        record = {k: payload[k] for k in ("Time", "Amount") if k in payload}
        for i in range(1, 29):
            key = f"V{i}"
            if key in payload:
                record[key] = payload[key]
        return pd.DataFrame([record])

    def _preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Run preprocessing and return a numpy array aligned with model features."""
        if self.preprocessor is not None:
            processed = self.preprocessor.transform(df.copy())
        else:
            processed = df.copy()

        processed = pd.DataFrame(processed)

        # Align column order with what the model was fitted on.
        if self.feature_names_:
            missing = [c for c in self.feature_names_ if c not in processed.columns]
            if missing:
                raise ValueError(
                    f"Preprocessed data is missing features expected by model: {missing}"
                )
            processed = processed[self.feature_names_]

        return processed.values

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict_single(self, payload: Dict[str, Any]) -> PredictionOutput:
        """Predict fraud for a single transaction supplied as a dict."""
        if not self.is_loaded:
            raise RuntimeError("Predictor not loaded. Call load() first.")

        raw = self._raw_frame(payload)
        X = self._preprocess(raw)

        proba = float(self.model.predict_proba(X)[0, 1])
        is_fraud = bool(proba >= self.threshold)

        return PredictionOutput(
            is_fraud=is_fraud,
            fraud_probability=round(proba, 6),
            risk_level=_risk_level(proba),
            threshold=self.threshold,
            features_used=int(X.shape[1]),
        )

    def predict_batch(
        self, df: pd.DataFrame
    ) -> Tuple[List[PredictionOutput], int, int]:
        """Predict fraud for every row in a DataFrame.

        Returns (results, fraud_count, legitimate_count).
        """
        if not self.is_loaded:
            raise RuntimeError("Predictor not loaded. Call load() first.")

        X = self._preprocess(df)
        probabilities = self.model.predict_proba(X)[:, 1]
        preds = (probabilities >= self.threshold).astype(int)

        results: List[PredictionOutput] = []
        for proba, pred in zip(probabilities, preds):
            results.append(
                PredictionOutput(
                    is_fraud=bool(pred),
                    fraud_probability=round(float(proba), 6),
                    risk_level=_risk_level(float(proba)),
                    threshold=self.threshold,
                    features_used=int(X.shape[1]),
                )
            )

        fraud_count = int(preds.sum())
        legitimate_count = int(len(preds) - fraud_count)
        return results, fraud_count, legitimate_count
