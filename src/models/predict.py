"""
predict.py — Model packaging: inference + ONNX export.

Two responsibilities:
  1. HeartDiseasePredictor  — lazy-loading inference class (sklearn or ONNX)
  2. export_to_onnx()       — converts best_model.joblib → best_model.onnx

Lazy loading means the model is NOT loaded at import time — only on first
prediction call. This keeps FastAPI startup fast and allows the same class
to be imported in tests without touching disk.

Usage (inference):
    from src.models.predict import HeartDiseasePredictor
    predictor = HeartDiseasePredictor()                # no I/O yet
    result = predictor.predict({"age": 52, "sex": 1, ...})
    result = predictor.predict_batch(df)

Usage (ONNX export):
    python -m src.models.predict --export
    # → writes models/best_model.onnx + models/onnx_validation_report.json

Usage (quick smoke test):
    python -m src.models.predict --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SKLEARN_MODEL_PATH = MODELS_DIR / "best_model.joblib"
ONNX_MODEL_PATH    = MODELS_DIR / "best_model.onnx"
SUMMARY_PATH       = MODELS_DIR / "best_model_summary.json"
# NOTE: preprocessor is embedded inside best_model.joblib as a Pipeline step.
# No separate preprocessor.joblib is needed for inference.

# ---------------------------------------------------------------------------
# Feature schema — single source of truth
# Must match preprocess.py column taxonomy exactly.
# ---------------------------------------------------------------------------
RAW_FEATURE_NAMES: list[str] = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

FEATURE_DTYPES: dict[str, str] = {
    "age":      "float",
    "sex":      "int",
    "cp":       "int",
    "trestbps": "float",
    "chol":     "float",
    "fbs":      "int",
    "restecg":  "int",
    "thalach":  "float",
    "exang":    "int",
    "oldpeak":  "float",
    "slope":    "int",
    "ca":       "float",   # nullable
    "thal":     "float",   # nullable
}

FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "age":      (1,   120),
    "sex":      (0,   1),
    "cp":       (1,   4),
    "trestbps": (50,  300),
    "chol":     (50,  700),
    "fbs":      (0,   1),
    "restecg":  (0,   2),
    "thalach":  (50,  250),
    "exang":    (0,   1),
    "oldpeak":  (0.0, 10.0),
    "slope":    (1,   3),
    "ca":       (0,   3),
    "thal":     (3,   7),
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
class ValidationError(ValueError):
    """Raised when input features fail schema validation."""


def _validate_input(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and coerce a raw feature dict.
    Returns a cleaned dict; raises ValidationError on bad input.
    """
    errors: list[str] = []
    cleaned: dict[str, Any] = {}

    for feat in RAW_FEATURE_NAMES:
        if feat not in data:
            if feat in ("ca", "thal"):
                cleaned[feat] = float("nan")
                continue
            errors.append(f"Missing required feature: '{feat}'")
            continue

        val = data[feat]

        if val is None or (isinstance(val, float) and np.isnan(val)):
            if feat in ("ca", "thal"):
                cleaned[feat] = float("nan")
                continue
            else:
                errors.append(f"'{feat}' cannot be null")
                continue

        try:
            dtype = FEATURE_DTYPES[feat]
            val = float(val) if dtype == "float" else int(float(val))
        except (TypeError, ValueError):
            errors.append(f"'{feat}' must be numeric, got: {type(data[feat]).__name__}")
            continue

        lo, hi = FEATURE_RANGES[feat]
        if not (lo <= val <= hi):
            errors.append(f"'{feat}' = {val} is outside valid range [{lo}, {hi}]")
            continue

        cleaned[feat] = val

    if errors:
        raise ValidationError("Input validation failed:\n  " + "\n  ".join(errors))

    return cleaned


# ---------------------------------------------------------------------------
# HeartDiseasePredictor
# ---------------------------------------------------------------------------
class HeartDiseasePredictor:
    """
    Lazy-loading predictor for the Heart Disease classifier.

    Supports two backends:
      - 'sklearn'  : loads best_model.joblib + preprocessor.joblib (default)
      - 'onnx'     : loads best_model.onnx + preprocessor.joblib (faster inference)

    The model and preprocessor are loaded on the first call to predict() or
    predict_batch(), not at __init__ time.

    Parameters
    ----------
    backend : str
        'sklearn' (default) or 'onnx'
    model_path : Path | None
        Override default model path.
    """

    def __init__(
        self,
        backend: str = "sklearn",
        model_path: Path | None = None,
    ) -> None:
        if backend not in ("sklearn", "onnx"):
            raise ValueError(f"backend must be 'sklearn' or 'onnx', got '{backend}'")

        self.backend = backend
        self._model_path = model_path or (ONNX_MODEL_PATH if backend == "onnx" else SKLEARN_MODEL_PATH)

        # Pipeline (preprocessor + classifier) loaded lazily on first predict call
        self._pipeline: Any = None       # sklearn Pipeline object
        self._onnx_session: Any = None   # onnxruntime InferenceSession

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load the Pipeline from disk. Called once on first use."""
        if self._pipeline is not None or self._onnx_session is not None:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self._model_path}.\n"
                "Run: python -m src.models.train"
            )

        # Always load the sklearn Pipeline — it contains the preprocessor
        # which is needed even for the ONNX backend (to transform raw features).
        sklearn_pipeline_path = SKLEARN_MODEL_PATH
        if not sklearn_pipeline_path.exists():
            raise FileNotFoundError(
                f"sklearn Pipeline not found at {sklearn_pipeline_path}.\n"
                "Run: python -m src.models.train"
            )
        log.info("Loading sklearn Pipeline from %s", sklearn_pipeline_path)
        self._pipeline = joblib.load(sklearn_pipeline_path)

        if self.backend == "onnx":
            try:
                import onnxruntime as rt  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime is required for ONNX backend.\n"
                    "Install: pip install onnxruntime"
                ) from exc
            log.info("Loading ONNX session from %s", self._model_path)
            self._onnx_session = rt.InferenceSession(
                str(self._model_path),
                providers=["CPUExecutionProvider"],
            )

        log.info("Predictor ready (backend=%s)", self.backend)

    # ------------------------------------------------------------------
    # Internal predict
    # ------------------------------------------------------------------
    def _run_sklearn(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Run full Pipeline (preprocessor + classifier) on a raw feature DataFrame."""
        preds = self._pipeline.predict(df)
        probs = self._pipeline.predict_proba(df)[:, 1]
        return preds, probs

    def _run_onnx(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sess = self._onnx_session
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        prob_name = sess.get_outputs()[1].name

        outputs = sess.run([label_name, prob_name], {input_name: X})
        preds = np.array(outputs[0])
        raw_probs = outputs[1]

        # Handle three possible output shapes from skl2onnx:
        #   - list of dicts  [{0: p0, 1: p1}, ...]  (zipmap=True)
        #   - 2-D array      shape (N, 2)            (zipmap=False, most models)
        #   - 1-D array      shape (N,)              (some linear models)
        if isinstance(raw_probs, list) and isinstance(raw_probs[0], dict):
            probs = np.array([p[1] for p in raw_probs], dtype=np.float32)
        else:
            raw_probs = np.array(raw_probs, dtype=np.float32)
            if raw_probs.ndim == 2:
                probs = raw_probs[:, 1]   # column 1 = P(disease)
            else:
                probs = raw_probs         # already (N,)

        return preds, probs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Predict heart disease for a single patient.

        Parameters
        ----------
        features : dict
            Raw feature values keyed by feature name.
            Example:
              {"age": 52, "sex": 1, "cp": 4, "trestbps": 125, "chol": 212,
               "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
               "oldpeak": 1.0, "slope": 2, "ca": 2.0, "thal": 7.0}

        Returns
        -------
        dict with keys:
            prediction   : int   — 0 = no disease, 1 = disease
            probability  : float — probability of disease (class 1)
            risk_label   : str   — "Low" / "Moderate" / "High"
            latency_ms   : float — inference time in milliseconds
        """
        self._load()
        cleaned = _validate_input(features)
        df = pd.DataFrame([cleaned], columns=RAW_FEATURE_NAMES)

        t0 = time.perf_counter()
        if self.backend == "onnx":
            # ONNX needs preprocessed float32 array — extract preprocessor from pipeline
            X = self._pipeline.named_steps["preprocessor"].transform(df).astype("float32")
            preds, probs = self._run_onnx(X)
        else:
            preds, probs = self._run_sklearn(df)
        latency_ms = (time.perf_counter() - t0) * 1000

        prob = float(probs[0])
        pred = int(preds[0])
        risk = "High" if prob >= 0.65 else ("Moderate" if prob >= 0.35 else "Low")

        return {
            "prediction": pred,
            "probability": round(prob, 4),
            "risk_label": risk,
            "latency_ms": round(latency_ms, 3),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict for a DataFrame of patients.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all 13 raw feature columns.
            Extra columns (e.g. 'target') are ignored.

        Returns
        -------
        pd.DataFrame with three new columns: prediction, probability, risk_label
        """
        self._load()

        raw_cols = [c for c in RAW_FEATURE_NAMES if c in df.columns]
        input_df = df[raw_cols].copy()

        t0 = time.perf_counter()
        if self.backend == "onnx":
            X = self._pipeline.named_steps["preprocessor"].transform(input_df).astype("float32")
            preds, probs = self._run_onnx(X)
        else:
            preds, probs = self._run_sklearn(input_df)
        latency_ms = (time.perf_counter() - t0) * 1000

        result = df.copy()
        result["prediction"] = preds.astype(int)
        result["probability"] = np.round(probs.astype(float), 4)
        result["risk_label"] = pd.cut(
            result["probability"],
            bins=[-0.001, 0.35, 0.65, 1.001],
            labels=["Low", "Moderate", "High"],
        )

        log.info(
            "Batch prediction: %d rows in %.1f ms (%.2f ms/row)",
            len(df), latency_ms, latency_ms / max(len(df), 1),
        )
        return result

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None or self._onnx_session is not None

    def model_info(self) -> dict:
        """Return metadata about the loaded model."""
        self._load()
        info: dict[str, Any] = {
            "backend": self.backend,
            "model_path": str(self._model_path),
            "pipeline_steps": (
                list(self._pipeline.named_steps.keys()) if self._pipeline else ["preprocessor", "classifier"]
            ),
        }
        if SUMMARY_PATH.exists():
            summary = json.loads(SUMMARY_PATH.read_text())
            info["model_name"] = summary.get("best_model")
            info["test_roc_auc"] = summary.get("test_metrics", {}).get("test_roc_auc")
            info["mlflow_run_id"] = summary.get("mlflow_run_id")
        return info


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_to_onnx(
    sklearn_model_path: Path = SKLEARN_MODEL_PATH,
    onnx_output_path: Path = ONNX_MODEL_PATH,
    n_features: int | None = None,
) -> Path:
    """
    Convert best_model.joblib to best_model.onnx.

    Steps:
      1. Load sklearn model + preprocessor
      2. Infer number of processed features from preprocessor output
      3. Convert with skl2onnx
      4. Validate ONNX predictions match sklearn on test data
      5. Save ONNX model + validation report JSON

    Returns path to the written ONNX file.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        raise ImportError(
            "skl2onnx is required for ONNX export.\n"
            "Install: pip install skl2onnx onnx"
        ) from exc

    log.info("=== ONNX Export ===")

    if not sklearn_model_path.exists():
        raise FileNotFoundError(f"Pipeline not found: {sklearn_model_path}\n"
                                "Run: python -m src.models.train")

    log.info("Loading Pipeline from %s", sklearn_model_path)
    pipeline = joblib.load(sklearn_model_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    # Infer feature dimension from preprocessor output
    if n_features is None:
        dummy = pd.DataFrame(
            [[52, 1, 4, 125, 212, 0, 1, 168, 0, 1.0, 2, 2.0, 7.0]],
            columns=RAW_FEATURE_NAMES,
        )
        transformed_dummy = preprocessor.transform(dummy)
        n_features = transformed_dummy.shape[1]
        log.info("Inferred processed feature count: %d", n_features)

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    log.info("Converting %s → ONNX (input shape: [N, %d])", type(model).__name__, n_features)

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=17,
        options={type(model): {"zipmap": False}},
    )

    # Validate against test data
    validation_report: dict[str, Any] = {"status": "skipped", "n_samples": 0}
    x_test_path = PROCESSED_DIR / "X_test.csv"
    y_test_path = PROCESSED_DIR / "y_test.csv"

    x_raw_path = PROCESSED_DIR / "X_train.csv"   # raw processed CSVs still exist
    if x_test_path.exists() and y_test_path.exists():
        log.info("Validating ONNX output against sklearn predictions...")
        import io
        import onnxruntime as rt  # type: ignore

        # X_test.csv contains already-preprocessed features (from preprocess.py output)
        X_test = pd.read_csv(x_test_path).values.astype(np.float32)

        sk_preds = model.predict(X_test)
        sk_probs = model.predict_proba(X_test)[:, 1]

        buf = io.BytesIO(onnx_model.SerializeToString())
        sess = rt.InferenceSession(buf.read(), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        onnx_out = sess.run(None, {input_name: X_test})
        onnx_preds = np.array(onnx_out[0])
        raw = np.array(onnx_out[1], dtype=np.float32)
        onnx_probs = raw[:, 1] if raw.ndim == 2 else raw

        pred_match = float(np.mean(sk_preds == onnx_preds))
        prob_max_diff = float(np.max(np.abs(sk_probs - onnx_probs)))

        validation_report = {
            "status": "passed" if pred_match == 1.0 and prob_max_diff < 1e-4 else "warning",
            "n_samples": len(X_test),
            "prediction_match_rate": round(pred_match, 6),
            "probability_max_diff": round(prob_max_diff, 8),
            "sklearn_model_type": type(model).__name__,
            "n_features": n_features,
            "target_opset": 17,
        }
        log.info(
            "Validation: pred_match=%.4f  prob_max_diff=%.2e  status=%s",
            pred_match, prob_max_diff, validation_report["status"],
        )
    else:
        log.warning("Test data not found — skipping ONNX validation.")

    # Save
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(onnx_output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    log.info(
        "ONNX model saved → %s  (%.1f KB)",
        onnx_output_path,
        onnx_output_path.stat().st_size / 1024,
    )

    report_path = MODELS_DIR / "onnx_validation_report.json"
    report_path.write_text(json.dumps(validation_report, indent=2))
    log.info("Validation report → %s", report_path)

    return onnx_output_path


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    """End-to-end check: sklearn predict → ONNX export → ONNX predict → batch."""
    SAMPLE = {
        "age": 52.0, "sex": 1, "cp": 4, "trestbps": 125.0,
        "chol": 212.0, "fbs": 0, "restecg": 1, "thalach": 168.0,
        "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2.0, "thal": 7.0,
    }

    print("\n── Smoke Test ───────────────────────────────────────────")

    print("\n[1] sklearn backend")
    sk_pred = HeartDiseasePredictor(backend="sklearn")
    result = sk_pred.predict(SAMPLE)
    for k, v in result.items():
        print(f"    {k:<14}: {v}")
    print(f"    model_info    : {sk_pred.model_info()}")

    print("\n[2] ONNX export")
    onnx_path = export_to_onnx()
    print(f"    Written to    : {onnx_path}")

    print("\n[3] ONNX backend")
    onnx_pred = HeartDiseasePredictor(backend="onnx")
    result_onnx = onnx_pred.predict(SAMPLE)
    for k, v in result_onnx.items():
        print(f"    {k:<14}: {v}")

    print("\n[4] Batch prediction (5 rows)")
    df = pd.DataFrame([SAMPLE] * 5)
    batch_result = HeartDiseasePredictor().predict_batch(df)
    print(batch_result[["prediction", "probability", "risk_label"]].to_string())

    print("\n── Smoke Test PASSED ────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heart Disease model packaging.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", action="store_true",
                       help="Export best_model.joblib to best_model.onnx")
    group.add_argument("--smoke-test", action="store_true",
                       help="Run end-to-end smoke test (sklearn + ONNX)")
    args = parser.parse_args()

    if args.export:
        export_to_onnx()
    elif args.smoke_test:
        _smoke_test()