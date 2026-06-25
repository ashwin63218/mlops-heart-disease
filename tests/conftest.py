"""
conftest.py — Shared pytest fixtures for the MLOps Heart Disease test suite.

Fixtures are scoped to minimise I/O:
  - 'processed_dir' and 'models_dir' are session-scoped (resolved once).
  - 'sample_patient' is function-scoped (cheap dict, fine to recreate).
  - 'predictor' is session-scoped (model loaded once across all test_model tests).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Project root — works regardless of where pytest is invoked from
# ---------------------------------------------------------------------------
def _find_project_root() -> Path:
    candidate = Path(__file__).resolve().parent
    for _ in range(5):
        if (candidate / "data").is_dir() or (candidate / "src").is_dir():
            return candidate
        candidate = candidate.parent
    raise FileNotFoundError("Could not locate project root (no data/ or src/ dir found).")


PROJECT_ROOT = _find_project_root()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Directory fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def processed_dir() -> Path:
    if not PROCESSED_DIR.exists():
        pytest.skip("Processed data not found — run preprocess.py first.")
    return PROCESSED_DIR


@pytest.fixture(scope="session")
def models_dir() -> Path:
    if not MODELS_DIR.exists():
        pytest.skip("Models directory not found — run train.py first.")
    return MODELS_DIR


# ---------------------------------------------------------------------------
# Processed data fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def X_train(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "X_train.csv"
    if not path.exists():
        pytest.skip(f"X_train.csv not found at {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def X_test(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "X_test.csv"
    if not path.exists():
        pytest.skip(f"X_test.csv not found at {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def y_train(processed_dir: Path) -> pd.Series:
    path = processed_dir / "y_train.csv"
    if not path.exists():
        pytest.skip(f"y_train.csv not found at {path}")
    return pd.read_csv(path).squeeze()


@pytest.fixture(scope="session")
def y_test(processed_dir: Path) -> pd.Series:
    path = processed_dir / "y_test.csv"
    if not path.exists():
        pytest.skip(f"y_test.csv not found at {path}")
    return pd.read_csv(path).squeeze()


# ---------------------------------------------------------------------------
# Sample patient records
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_patient() -> dict:
    """A single valid patient record (all 13 raw features)."""
    return {
        "age": 52.0, "sex": 1, "cp": 4, "trestbps": 125.0,
        "chol": 212.0, "fbs": 0, "restecg": 1, "thalach": 168.0,
        "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2.0, "thal": 7.0,
    }


@pytest.fixture
def sample_patient_no_disease() -> dict:
    """A patient profile typically associated with no disease."""
    return {
        "age": 35.0, "sex": 0, "cp": 1, "trestbps": 110.0,
        "chol": 180.0, "fbs": 0, "restecg": 0, "thalach": 185.0,
        "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0.0, "thal": 3.0,
    }


@pytest.fixture
def sample_batch_df(sample_patient, sample_patient_no_disease) -> pd.DataFrame:
    """A small DataFrame with two contrasting patient records."""
    return pd.DataFrame([sample_patient, sample_patient_no_disease])


# ---------------------------------------------------------------------------
# Predictor fixture (session-scoped — loads model once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def predictor():
    """Lazy-loaded HeartDiseasePredictor (sklearn backend)."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.models.predict import HeartDiseasePredictor

    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        pytest.skip("best_model.joblib not found — run train.py first.")

    return HeartDiseasePredictor(backend="sklearn")