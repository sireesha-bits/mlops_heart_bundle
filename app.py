
# app.py
"""
Heart Disease Predictor API (FastAPI)

Features:
- /predict: Single-record prediction (editable example in Swagger); optional 'target' for correctness metrics
- /predict_batch: CSV upload for batch predictions; optional 'target' column for correctness metrics
- /health: Health check with query parameters to include model info, metrics summary, and version
- /metrics: Prometheus endpoint; supports format=prometheus|json and optional endpoint filter for JSON summary
- CORS enabled
- Structured request logging to console and rotatable file

Run:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from typing import Optional, List
import logging
import sys
import time
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware


# ------------------------------------------------------------------------------
# Structured logging (console + rotatable file) and Uvicorn coordination
# ------------------------------------------------------------------------------
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
))
logger.addHandler(console)

Path("logs").mkdir(exist_ok=True)
from logging.handlers import RotatingFileHandler
file_handler = RotatingFileHandler("logs/app.log", maxBytes=10_000_000, backupCount=5)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
))
logger.addHandler(file_handler)

# Let Uvicorn reuse our handlers (avoid duplicate or differently formatted output)
for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    uvlog = logging.getLogger(name)
    uvlog.handlers = []
    uvlog.propagate = True
    uvlog.setLevel(logging.INFO)

class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        logger.info(f"REQ {request.method} {request.url.path} from {request.client.host}")
        response = await call_next(request)
        dur_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"RES {request.method} {request.url.path} -> {response.status_code} in {dur_ms:.1f} ms"
        )
        return response


# ------------------------------------------------------------------------------
# App metadata (appears in /docs)
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Heart Disease Predictor API",
    description=(
        "Predict heart disease risk using a trained ML model.\n\n"
        "**How to use:**\n"
        "- Open **`/docs`** and try **POST /predict** (editable example pre-filled).\n"
        "- Adjust **query params** like `threshold` (0..1) and `return_proba`.\n"
        "- Use **POST /predict_batch** to upload a CSV for batch scoring.\n"
        "- Optionally include `target` to track **correct vs incorrect** predictions in `/metrics`.\n"
        "- Health and metrics support query parameters for richer status/summary."
    ),
    version="1.0.0",
)
app.add_middleware(RequestLogMiddleware)

# ------------------------------------------------------------------------------
# CORS (allow your web app to call the API)
# ------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 Restrict to your frontend origin(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Model loading (safe startup)
# ------------------------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
model = None

@app.on_event("startup")
def load_model():
    """Load the trained pipeline safely; if missing, start and return 503 on /predict."""
    global model
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            logger.info("Model loaded from %s", MODEL_PATH)
        else:
            logger.warning("Model not found at %s; /predict will return 503 until trained", MODEL_PATH)
            model = None
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        model = None

# ------------------------------------------------------------------------------
# Metrics: requests, latency, and correctness counters
# ------------------------------------------------------------------------------
req_counter = Counter("api_requests_total", "Total API requests", ["endpoint"])
latency = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])

pred_correct = Counter(
    "prediction_correct_total",
    "Number of predictions where predicted == target",
    ["endpoint"],
)
pred_incorrect = Counter(
    "prediction_incorrect_total",
    "Number of predictions where predicted != target",
    ["endpoint"],
)

# In-memory mirrors (for quick JSON summaries /health, /metrics?format=json)
_INMEM = defaultdict(lambda: {"correct": 0, "incorrect": 0})
_REQS = defaultdict(int)  # total requests per endpoint (in-memory)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    endpoint = request.url.path
    _REQS[endpoint] += 1
    req_counter.labels(endpoint=endpoint).inc()
    with latency.labels(endpoint=endpoint).time():
        response = await call_next(request)
    return response

def _summarize_correctness() -> dict:
    return {
        "predict": {
            "correct": _INMEM["/predict"]["correct"],
            "incorrect": _INMEM["/predict"]["incorrect"],
        },
        "predict_batch": {
            "correct": _INMEM["/predict_batch"]["correct"],
            "incorrect": _INMEM["/predict_batch"]["incorrect"],
        },
    }

# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------
class Input(BaseModel):
    """Single-record input with optional label `target` for correctness metrics."""
    age: float      = Field(..., ge=0, le=120, description="Age (years)")
    sex: int        = Field(..., ge=0, le=1, description="Sex (1 = male; 0 = female)")
    cp: int         = Field(..., ge=1, le=4, description="Chest pain type (1..4)")
    trestbps: float = Field(..., ge=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: float     = Field(..., ge=0, le=700, description="Serum cholesterol (mg/dL)")
    fbs: int        = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dL (1/0)")
    restecg: int    = Field(..., ge=0, le=2, description="Resting ECG results (0..2)")
    thalach: float  = Field(..., ge=0, le=300, description="Max heart rate achieved")
    exang: int      = Field(..., ge=0, le=1, description="Exercise-induced angina (1/0)")
    oldpeak: float  = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int      = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0..2)")
    ca: int         = Field(..., ge=0, le=3, description="Number of major vessels colored by fluoroscopy (0..3)")
    thal: int       = Field(..., ge=0, le=7, description="Thalassemia code (e.g., 3=normal; 6=fixed; 7=reversible)")

    target: Optional[int] = Field(
        None, ge=0, le=1,
        description="Optional ground-truth label (0=healthy, 1=disease). If provided, metrics will count correct/incorrect."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
                "target": 0
            }
        }

class PredictionOut(BaseModel):
    prediction: int = Field(..., description="Predicted class: 0=no disease, 1=disease")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Probability of disease (if available)")
    threshold: float = Field(..., ge=0, le=1, description="Decision threshold applied")

class BatchPredictionOut(BaseModel):
    predictions: List[int] = Field(..., description="Predicted class per row")
    confidences: Optional[List[float]] = Field(None, description="Probabilities per row (if available)")
    threshold: float = Field(..., ge=0, le=1)
    rows: int = Field(..., description="Number of rows processed")
    errors: Optional[List[str]] = Field(None, description="Row-level errors (if any)")

FEATURE_COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

# ------------------------------------------------------------------------------
# /predict (single record)
# ------------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionOut, summary="Predict heart disease risk")
async def predict(
    payload: Input = Body(
        ...,
        examples={
            "Typical Example": {
                "summary": "Healthy-ish example",
                "value": {
                    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1, "target": 0
                },
            },
            "High Risk Example": {
                "summary": "Try higher risk inputs",
                "value": {
                    "age": 68, "sex": 1, "cp": 4, "trestbps": 160, "chol": 290,
                    "fbs": 1, "restecg": 2, "thalach": 120, "exang": 1,
                    "oldpeak": 4.2, "slope": 2, "ca": 2, "thal": 7, "target": 1
                },
            },
        },
    ),
    threshold: float = Query(
        0.5, ge=0, le=1,
        description="Decision threshold for positive class (0..1). Try 0.3 or 0.7 to see behavior change."
    ),
    return_proba: bool = Query(
        True,
        description="Include probability in response (true/false)."
    ),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first (run src/train.py).")

    row = {c: getattr(payload, c) for c in FEATURE_COLUMNS}
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    try:
        y_hat = int(model.predict(X)[0])
        try:
            proba = float(model.predict_proba(X)[0][1])
        except Exception:
            proba = None

        if proba is not None:
            y_hat = 1 if proba >= threshold else 0

        if payload.target is not None:
            true_label = int(payload.target)
            if y_hat == true_label:
                pred_correct.labels(endpoint="/predict").inc()
                _INMEM["/predict"]["correct"] += 1
            else:
                pred_incorrect.labels(endpoint="/predict").inc()
                _INMEM["/predict"]["incorrect"] += 1

        return PredictionOut(
            prediction=y_hat,
            confidence=(proba if return_proba else None),
            threshold=threshold
        )
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

# ------------------------------------------------------------------------------
# /predict_batch (CSV upload)
# ------------------------------------------------------------------------------
@app.post("/predict_batch", response_model=BatchPredictionOut, summary="Batch predictions from CSV upload")
async def predict_batch(
    file: UploadFile = File(..., description="CSV with columns: " + ", ".join(FEATURE_COLUMNS) + " [+ optional 'target']"),
    threshold: float = Query(0.5, ge=0, le=1, description="Decision threshold for classification"),
    return_proba: bool = Query(True, description="Include probabilities in response"),
    has_header: bool = Query(True, description="CSV contains a header row with column names"),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first (run src/train.py).")

    try:
        df = pd.read_csv(file.file, header=0 if has_header else None)
        if not has_header:
            df.columns = FEATURE_COLUMNS
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}. Expected: {FEATURE_COLUMNS} [+ optional 'target']"
        )

    try:
        X = df[FEATURE_COLUMNS]
        preds = model.predict(X)

        confidences = None
        try:
            confidences = model.predict_proba(X)[:, 1].tolist()
        except Exception:
            confidences = None

        if confidences is not None:
            preds = [1 if p >= threshold else 0 for p in confidences]

        if "target" in df.columns:
            valid_mask = df["target"].isin([0, 1])
            valid_targets = df.loc[valid_mask, "target"].astype(int).tolist()
            valid_preds = [int(p) for idx, p in enumerate(preds) if valid_mask.iloc[idx]]

            correct = sum(int(p == t) for p, t in zip(valid_preds, valid_targets))
            incorrect = len(valid_preds) - correct

            if correct > 0:
                pred_correct.labels(endpoint="/predict_batch").inc(correct)
                _INMEM["/predict_batch"]["correct"] += int(correct)
            if incorrect > 0:
                pred_incorrect.labels(endpoint="/predict_batch").inc(incorrect)
                _INMEM["/predict_batch"]["incorrect"] += int(incorrect)

        return BatchPredictionOut(
            predictions=[int(p) for p in preds],
            confidences=(confidences if return_proba else None),
            threshold=threshold,
            rows=int(len(df)),
        )
    except Exception as e:
        logger.exception("Batch prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch inference error: {e}")

# ------------------------------------------------------------------------------
# Health & Metrics endpoints
# ------------------------------------------------------------------------------
@app.get("/health", summary="Health check (with optional details)")
def health(
    include_model: bool = Query(False, description="Include model path & loaded status"),
    include_metrics_summary: bool = Query(False, description="Include correctness/request summaries"),
    include_version: bool = Query(False, description="Include API & Python version info"),
):
    payload = {"status": "ok", "timestamp": int(time.time())}

    if include_model:
        payload["model"] = {
            "loaded": bool(model is not None),
            "path": str(MODEL_PATH),
        }

    if include_metrics_summary:
        payload["metrics_summary"] = {
            "correctness": _summarize_correctness(),
            "requests": dict(_REQS),
            "note": "In-memory mirrors; Prometheus holds the source of truth.",
        }

    if include_version:
        payload["version"] = {
            "api": getattr(app, "version", "unknown"),
            "python": sys.version,
        }

    return payload

@app.get("/metrics", summary="Prometheus metrics (or JSON summary)")
def metrics(
    format: str = Query("prometheus", description="Return 'prometheus' plaintext (default) or 'json' summary."),
    filter_endpoint: Optional[str] = Query(None, description="Filter JSON summary to '/predict' or '/predict_batch'."),
):
    fmt = format.strip().lower()
    if fmt == "prometheus":
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    summary = {
        "correctness": _summarize_correctness(),
        "requests": dict(_REQS),
        "endpoints": ["/predict", "/predict_batch", "/health", "/metrics"],
        "note": "JSON view for quick checks; not scraped by Prometheus.",
    }
    if filter_endpoint:
        key = filter_endpoint.strip()
        if key not in ["/predict", "/predict_batch"]:
            return {"error": f"Unknown endpoint filter: {filter_endpoint}", "hint": "Use '/predict' or '/predict_batch'."}
        summary["correctness"] = {key.strip("/"): summary["correctness"][key.strip("/")]}

    return summary
