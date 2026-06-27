"""
app.py — FastAPI inference service for Heart Disease Classification.

Endpoints:
    GET  /health   — liveness probe (Kubernetes)
    GET  /ready    — readiness probe (checks model is loaded)
    GET  /info     — model metadata (name, AUC, MLflow run ID)
    POST /predict  — single patient prediction
    POST /predict/batch — batch prediction (up to 100 rows)

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

    Type http://127.0.0.1:8000/docs and explore the API with Swagger UI.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Path setup — allows running from project root or inside src/api/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import (  # noqa: E402
    HeartDiseasePredictor,
    ValidationError,
    RAW_FEATURE_NAMES,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Global predictor — initialised once at startup, reused across all requests
# ---------------------------------------------------------------------------
_BACKEND = os.getenv("MODEL_BACKEND", "sklearn")  # override with ONNX in prod
predictor = HeartDiseasePredictor(backend=_BACKEND)

# ---------------------------------------------------------------------------
# App startup / shutdown
# ---------------------------------------------------------------------------
_startup_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the predictor so the first real request is not slow."""
    global _startup_time
    instrumentator.expose(app)
    t0 = time.perf_counter()
    log.info("Warming up predictor (backend=%s)...", _BACKEND)
    try:
        predictor._load()
        _startup_time = time.perf_counter() - t0
        log.info("Predictor ready in %.2fs", _startup_time)
    except FileNotFoundError as exc:
        log.error("Model not found at startup: %s", exc)
        log.error("Run python -m src.models.train before starting the API.")
    yield
    log.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Heart Disease Classifier API",
    description=(
        "MLOps Assignment — AIMLCZG523\n\n"
        "Predicts presence of heart disease from 13 clinical features.\n"
        "Dataset: UCI Heart Disease (Cleveland + multi-source merge)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Prometheus metrics — exposed at /metrics
# instrument() registers middleware; expose() adds the /metrics route
instrumentator = Instrumentator().instrument(app)


@app.get("/", tags=["ops"])
def read_root():
    """Root endpoint to welcome users and point to documentation."""
    return {
        "message": "Welcome to the Heart Disease Classifier API!",
        "documentation": "/docs",
        "health_check": "/health",
    }


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PatientFeatures(BaseModel):
    """
    Raw clinical features for a single patient.
    All values must fall within clinically valid ranges.
    """

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex: 0=Female, 1=Male")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure (mmHg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120: 0/1")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina: 0/1")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression (exercise vs rest)")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST (1-3)")
    ca: Optional[float] = Field(
        None, ge=0, le=3, description="Number of major vessels (0-3); null allowed"
    )
    thal: Optional[float] = Field(
        None, ge=3, le=7, description="Thal: 3=Normal, 6=Fixed defect, 7=Reversible; null allowed"
    )

    @field_validator("ca", "thal", mode="before")
    @classmethod
    def allow_null(cls, v):
        """Accept None, empty string, or NaN as missing values for nullable fields."""
        if v == "" or v != v:  # NaN check: NaN != NaN
            return None
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 52,
                "sex": 1,
                "cp": 4,
                "trestbps": 125,
                "chol": 212,
                "fbs": 0,
                "restecg": 1,
                "thalach": 168,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 2,
                "ca": 2,
                "thal": 7,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = No Disease, 1 = Disease Present")
    confidence: float = Field(..., description="Model confidence (probability of prediction)")
    probability: float = Field(..., description="Raw probability of disease (class 1)")
    risk_label: str = Field(..., description="Low / Moderate / High")
    latency_ms: float = Field(..., description="Inference time in milliseconds")


class BatchRequest(BaseModel):
    patients: list[PatientFeatures] = Field(
        ..., min_length=1, max_length=100, description="List of patient records (max 100)"
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[dict[str, Any]]
    count: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    backend: str
    uptime_s: float


# ---------------------------------------------------------------------------
# Middleware — log every request with latency
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    log.info(
        "%s %s → %d  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc), "type": "ValidationError"},
    )


@app.exception_handler(FileNotFoundError)
async def model_not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "Model not loaded. Run train.py first.",
            "type": "ModelNotFound",
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["ops"],
)
def health():
    """
    Kubernetes liveness probe.
    Returns 200 as long as the process is running.
    Does NOT check whether the model is loaded — use /ready for that.
    """
    return {"status": "ok", "version": app.version}


@app.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    tags=["ops"],
)
def ready():
    """
    Kubernetes readiness probe.
    Returns 200 only when the model is loaded and ready to serve traffic.
    Returns 503 if the model failed to load at startup.
    """
    loaded = predictor.is_loaded
    if not loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not yet loaded.",
        )
    return {
        "status": "ready",
        "model_loaded": loaded,
        "backend": predictor.backend,
        "uptime_s": round(time.perf_counter() - _startup_time, 1),
    }


@app.get(
    "/info",
    summary="Model metadata",
    tags=["ops"],
)
def info():
    """Returns model name, test AUC, MLflow run ID, and pipeline steps."""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )
    return predictor.model_info()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict heart disease for a single patient",
    tags=["inference"],
    status_code=status.HTTP_200_OK,
)
def predict(patient: PatientFeatures):
    """
    Accepts 13 raw clinical feature values and returns:
    - `prediction`  : 0 (No Disease) or 1 (Disease Present)
    - `confidence`  : probability of the predicted class
    - `probability` : raw P(disease=1) — useful for threshold tuning
    - `risk_label`  : Low / Moderate / High
    - `latency_ms`  : server-side inference time
    """
    features = patient.model_dump()

    try:
        result = predictor.predict(features)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    prob = result["probability"]
    pred = result["prediction"]

    return {
        "prediction": pred,
        "confidence": round(prob if pred == 1 else 1 - prob, 4),
        "probability": prob,
        "risk_label": result["risk_label"],
        "latency_ms": result["latency_ms"],
    }


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict heart disease for a batch of patients",
    tags=["inference"],
    status_code=status.HTTP_200_OK,
)
def predict_batch(batch: BatchRequest):
    """
    Accepts a list of up to 100 patient records.
    Returns predictions for all patients in a single call.
    """
    import pandas as pd

    records = [p.model_dump() for p in batch.patients]
    df = pd.DataFrame(records, columns=RAW_FEATURE_NAMES)

    t0 = time.perf_counter()
    try:
        result_df = predictor.predict_batch(df)
    except (ValidationError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    predictions = []
    for _, row in result_df.iterrows():
        pred = int(row["prediction"])
        prob = float(row["probability"])
        predictions.append(
            {
                "prediction": pred,
                "confidence": round(prob if pred == 1 else 1 - prob, 4),
                "probability": prob,
                "risk_label": str(row["risk_label"]),
            }
        )

    return {
        "predictions": predictions,
        "count": len(predictions),
        "latency_ms": latency_ms,
    }
