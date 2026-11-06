"""
Pydantic schemas for the Fraud Detection API.
Defines request and response models for single and batch prediction.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class TransactionInput(BaseModel):
    """Single transaction input for fraud prediction.

    Matches the credit card fraud dataset schema:
    - V1..V28: PCA-transformed features (anonymized)
    - Time: seconds elapsed since the first transaction in the dataset
    - Amount: transaction amount
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Time": 0.0,
                "Amount": 149.62,
                **{f"V{i}": round(-1.0 + (i * 0.1), 4) for i in range(1, 29)},
            }
        }
    )

    Time: float = Field(..., description="Seconds elapsed since first transaction")
    Amount: float = Field(..., description="Transaction amount")
    # V1..V28 PCA features
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class PredictionOutput(BaseModel):
    """Result of a single fraud prediction."""

    is_fraud: bool = Field(..., description="Whether the transaction is predicted as fraud")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of fraud (0-1)")
    risk_level: str = Field(..., description="Risk band: Low / Medium / High / Critical")
    threshold: float = Field(..., description="Decision threshold used")
    features_used: int = Field(..., description="Number of features fed to the model")


class BatchPredictionOutput(BaseModel):
    """Aggregated result of a batch prediction run."""

    total_transactions: int
    fraud_count: int
    legitimate_count: int
    fraud_rate: float
    threshold: float
    results: List[PredictionOutput]
    download_url: Optional[str] = None


class ModelInfo(BaseModel):
    """Metadata about the deployed model."""

    model_type: str
    features_used: int
    feature_names: List[str]
    threshold: float
    metrics: Optional[dict] = None
    version: str = "1.0.0"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"
