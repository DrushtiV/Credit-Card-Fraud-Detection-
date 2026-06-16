"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Credit Card Fraud Detection — FastAPI Inference Server                     ║
║  Run:  uvicorn api:app --reload --port 8000                                 ║
║  Docs: http://localhost:8000/docs                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import joblib, os, json

# ── Load artefacts ────────────────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
SCALER    = joblib.load(os.path.join(BASE, 'models', 'scaler.pkl'))
MODEL     = joblib.load(os.path.join(BASE, 'models', 'best_model_Logistic_Regression.pkl'))
METRICS   = json.load(open(os.path.join(BASE, 'outputs', 'metrics.json')))
FEAT_COLS = [f'V{i}' for i in range(1, 29)] + ['Amount']

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="💳 Credit Card Fraud Detection API",
    description=(
        "Real-time fraud scoring for credit card transactions.\n\n"
        "**Model**: Logistic Regression trained on SMOTE-balanced data.\n"
        "**Features**: V1–V28 (PCA-transformed) + log-scaled Amount.\n"
        "**ROC-AUC**: 1.0000  |  **F1**: 0.9751  |  **Recall**: 1.0000"
    ),
    version="1.0.0",
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class Transaction(BaseModel):
    """Single credit card transaction."""
    V1:  float = Field(..., example=-1.36)
    V2:  float = Field(..., example=-0.07)
    V3:  float = Field(..., example=2.54)
    V4:  float = Field(..., example=1.38)
    V5:  float = Field(..., example=-0.34)
    V6:  float = Field(..., example=0.46)
    V7:  float = Field(..., example=0.24)
    V8:  float = Field(..., example=0.10)
    V9:  float = Field(..., example=0.36)
    V10: float = Field(..., example=0.09)
    V11: float = Field(..., example=-0.55)
    V12: float = Field(..., example=-0.62)
    V13: float = Field(..., example=-0.99)
    V14: float = Field(..., example=-0.31)
    V15: float = Field(..., example=1.47)
    V16: float = Field(..., example=-0.47)
    V17: float = Field(..., example=0.21)
    V18: float = Field(..., example=0.03)
    V19: float = Field(..., example=0.40)
    V20: float = Field(..., example=0.25)
    V21: float = Field(..., example=-0.02)
    V22: float = Field(..., example=0.28)
    V23: float = Field(..., example=-0.11)
    V24: float = Field(..., example=0.07)
    V25: float = Field(..., example=0.13)
    V26: float = Field(..., example=-0.19)
    V27: float = Field(..., example=0.13)
    V28: float = Field(..., example=-0.02)
    Amount: float = Field(..., ge=0, example=149.62)

class PredictionResponse(BaseModel):
    is_fraud:       bool
    fraud_probability: float
    risk_level:     str   # LOW / MEDIUM / HIGH / CRITICAL
    threshold_used: float
    message:        str

class BatchRequest(BaseModel):
    transactions: List[Transaction]
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)

class BatchResponse(BaseModel):
    total:    int
    fraud:    int
    legit:    int
    results:  List[PredictionResponse]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _to_features(txn: Transaction) -> np.ndarray:
    """Convert a Transaction to a scaled feature vector."""
    raw = np.array([[getattr(txn, f) if f != 'Amount' else np.log1p(txn.Amount)
                     for f in FEAT_COLS]])
    return SCALER.transform(raw)

def _risk_label(prob: float) -> str:
    if prob < 0.30: return "LOW"
    if prob < 0.60: return "MEDIUM"
    if prob < 0.85: return "HIGH"
    return "CRITICAL"

def _score(txn: Transaction, threshold: float = 0.5) -> PredictionResponse:
    X = _to_features(txn)
    prob       = float(MODEL.predict_proba(X)[0, 1])
    is_fraud   = prob >= threshold
    risk       = _risk_label(prob)
    msg_map    = {
        "LOW"     : "Transaction appears legitimate.",
        "MEDIUM"  : "Elevated risk — manual review recommended.",
        "HIGH"    : "High fraud probability — flag for investigation.",
        "CRITICAL": "⚠️  Near-certain fraud — block immediately.",
    }
    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(prob, 6),
        risk_level=risk,
        threshold_used=threshold,
        message=msg_map[risk],
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "model": "Logistic Regression", "roc_auc": 1.0}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_loaded": MODEL is not None,
            "scaler_loaded": SCALER is not None}

@app.get("/metrics", tags=["Info"])
def get_metrics():
    """Return training metrics for all models."""
    return METRICS

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(txn: Transaction, threshold: float = 0.5):
    """
    Score a single transaction for fraud.

    - **threshold**: Classification cutoff (default 0.5). Lower = more sensitive.
    """
    if not 0 < threshold < 1:
        raise HTTPException(422, "threshold must be between 0 and 1")
    return _score(txn, threshold)

@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
def predict_batch(req: BatchRequest):
    """
    Score up to 1,000 transactions in a single request.
    """
    if len(req.transactions) > 1000:
        raise HTTPException(413, "Batch size must not exceed 1,000 transactions.")
    predictions = [_score(t, req.threshold) for t in req.transactions]
    fraud_count = sum(p.is_fraud for p in predictions)
    return BatchResponse(
        total=len(predictions),
        fraud=fraud_count,
        legit=len(predictions) - fraud_count,
        results=predictions,
    )

@app.get("/example", tags=["Info"])
def get_example():
    """Return an example fraudulent transaction payload for testing."""
    return {
        "V1": -3.04, "V2": -3.16, "V3": 1.09, "V4": 2.29, "V5": 0.61,
        "V6": -0.82, "V7": -0.42, "V8": 0.10, "V9": -0.37, "V10": -2.10,
        "V11": -2.35, "V12": -0.21, "V13": 0.61, "V14": -2.66, "V15": 0.21,
        "V16": -0.60, "V17": -1.52, "V18": -0.01, "V19": 0.06, "V20": -0.07,
        "V21": -0.36, "V22": -0.15, "V23": 0.07, "V24": -0.18, "V25": 0.34,
        "V26": -0.19, "V27": -0.38, "V28": 0.01, "Amount": 23.50
    }
