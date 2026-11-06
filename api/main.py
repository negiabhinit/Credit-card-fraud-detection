"""
Fraud Detection API - FastAPI application.

Endpoints:
    GET     /                - API info
    GET     /health          - Health check
    GET     /model-info      - Model metadata + metrics
    POST    /predict         - Single transaction fraud prediction
    POST    /predict/batch   - Batch prediction from uploaded CSV
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .predictor import FraudPredictor
from .schemas import (
    BatchPredictionOutput,
    HealthResponse,
    ModelInfo,
    PredictionOutput,
    TransactionInput,
)

logger = logging.getLogger("fraud-detection-api")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detector_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
PREDICTIONS_DIR = PROJECT_ROOT / "reports" / "api_predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# App & global predictor
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="REST API for real-time and batch credit card fraud prediction.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Optional[FraudPredictor] = None


@app.on_event("startup")
def _startup() -> None:
    global predictor
    threshold_env = os.getenv("FRAUD_THRESHOLD")
    threshold = float(threshold_env) if threshold_env else None
    predictor = FraudPredictor(
        model_path=str(MODEL_PATH),
        preprocessor_path=str(PREPROCESSOR_PATH),
        metrics_path=str(METRICS_PATH),
        threshold=threshold,
    )
    logger.info("API startup complete")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/", response_model=Dict[str, Any], tags=["General"])
def root() -> Dict[str, Any]:
    return {
        "name": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if (predictor and predictor.is_loaded) else "degraded",
        model_loaded=bool(predictor and predictor.is_loaded),
    )


@app.get("/model-info", response_model=ModelInfo, tags=["General"])
def model_info() -> ModelInfo:
    if not predictor or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    metrics: Optional[Dict[str, Any]] = None
    if METRICS_PATH.exists():
        try:
            import json

            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read metrics: %s", exc)
    return ModelInfo(
        model_type=type(predictor.model).__name__,
        features_used=len(predictor.feature_names_),
        feature_names=predictor.feature_names_,
        threshold=predictor.threshold,
        metrics=metrics,
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(transaction: TransactionInput) -> PredictionOutput:
    """Predict fraud for a single transaction."""
    if not predictor or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    try:
        return predictor.predict_single(transaction.model_dump())
    except Exception as exc:
        logger.exception("Single prediction failed")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {exc}",
        )


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)) -> BatchPredictionOutput:
    """Predict fraud for every transaction in an uploaded CSV file."""
    if not predictor or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are supported.",
        )

    try:
        import io

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as exc:
        logger.exception("Failed to parse uploaded CSV")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse CSV file: {exc}",
        )

    required = {"Time", "Amount"} | {f"V{i}" for i in range(1, 29)}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV is missing required columns: {sorted(missing)}",
        )

    try:
        results, fraud_count, legitimate_count = predictor.predict_batch(df)
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction failed: {exc}",
        )

    total = len(results)
    fraud_rate = fraud_count / total if total else 0.0

    out_rows = []
    for row, res in zip(df.itertuples(index=False), results):
        out_rows.append(
            {
                "Time": row.Time,
                "Amount": row.Amount,
                "fraud_probability": res.fraud_probability,
                "is_fraud": res.is_fraud,
                "risk_level": res.risk_level,
                "threshold": res.threshold,
            }
        )
    out_df = pd.DataFrame(out_rows)
    out_filename = f"predictions_{file.filename}"
    out_path = PREDICTIONS_DIR / out_filename
    out_df.to_csv(out_path, index=False)
    download_url = f"/predictions/{out_filename}"

    return BatchPredictionOutput(
        total_transactions=total,
        fraud_count=fraud_count,
        legitimate_count=legitimate_count,
        fraud_rate=round(fraud_rate, 4),
        threshold=predictor.threshold,
        results=results,
        download_url=download_url,
    )


@app.get("/predictions/{filename}", tags=["Prediction"])
def download_predictions(filename: str):
    """Download a previously generated batch predictions CSV."""
    safe_name = Path(filename).name
    file_path = PREDICTIONS_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=safe_name,
    )


# --------------------------------------------------------------------------- #
# Static frontend
# --------------------------------------------------------------------------- #
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


@app.get("/ui", include_in_schema=False)
def ui() -> FileResponse:
    """Serve the web frontend."""
    return FileResponse(str(WEB_DIR / "index.html"))

