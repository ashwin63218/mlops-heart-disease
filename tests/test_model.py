"""
test_model.py — Unit tests for model loading and inference.

Covers Step 5.1 requirements:
  ✅ Model loads without error
  ✅ Predicts correct output shape
  ✅ Probabilities are in [0, 1]
  ✅ Predictions are binary (0 or 1)
  ✅ Batch inference output shape matches input
  ✅ Input validation catches bad data
  ✅ Latency is within acceptable bounds
  ✅ ONNX export produces a valid model file
  ✅ ONNX predictions match sklearn predictions

Run:
    pytest tests/test_model.py -v
    pytest tests/test_model.py -v -k "not onnx"   # skip ONNX tests
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable — must come before src.* imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import (  # noqa: E402
    HeartDiseasePredictor,
    RAW_FEATURE_NAMES,
    ValidationError,
    _validate_input,
)


# ---------------------------------------------------------------------------
# Model loading tests
# ---------------------------------------------------------------------------
class TestModelLoading:
    def test_predictor_instantiates_without_io(self):
        """__init__ must not touch disk — model is loaded lazily."""
        p = HeartDiseasePredictor(backend="sklearn")
        assert not p.is_loaded

    def test_predictor_loads_on_first_predict(self, predictor, sample_patient):
        """Model should be loaded after the first predict() call."""
        _ = predictor.predict(sample_patient)
        assert predictor.is_loaded

    def test_model_info_returns_dict(self, predictor, sample_patient):
        _ = predictor.predict(sample_patient)
        info = predictor.model_info()
        assert isinstance(info, dict)
        assert "backend" in info
        assert "model_path" in info
        assert "pipeline_steps" in info

    def test_pipeline_steps_present(self, predictor, sample_patient):
        _ = predictor.predict(sample_patient)
        info = predictor.model_info()
        steps = info["pipeline_steps"]
        assert "preprocessor" in steps, f"Missing 'preprocessor' step: {steps}"
        assert "classifier" in steps, f"Missing 'classifier' step: {steps}"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend must be"):
            HeartDiseasePredictor(backend="tensorflow")


# ---------------------------------------------------------------------------
# Single prediction output tests
# ---------------------------------------------------------------------------
class TestSinglePrediction:
    def test_predict_returns_dict(self, predictor, sample_patient):
        result = predictor.predict(sample_patient)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, predictor, sample_patient):
        result = predictor.predict(sample_patient)
        required = {"prediction", "probability", "risk_label", "latency_ms"}
        assert required <= result.keys(), f"Missing keys: {required - result.keys()}"

    def test_prediction_is_binary(self, predictor, sample_patient):
        result = predictor.predict(sample_patient)
        assert result["prediction"] in (
            0,
            1,
        ), f"prediction must be 0 or 1, got {result['prediction']}"

    def test_probability_in_unit_interval(self, predictor, sample_patient):
        result = predictor.predict(sample_patient)
        prob = result["probability"]
        assert 0.0 <= prob <= 1.0, f"probability must be in [0, 1], got {prob}"

    def test_risk_label_valid(self, predictor, sample_patient):
        result = predictor.predict(sample_patient)
        assert result["risk_label"] in (
            "Low",
            "Moderate",
            "High",
        ), f"Unexpected risk_label: {result['risk_label']}"

    def test_risk_label_consistent_with_probability(self, predictor, sample_patient):
        """Risk label thresholds: Low < 0.35, Moderate 0.35–0.65, High >= 0.65."""
        result = predictor.predict(sample_patient)
        prob = result["probability"]
        label = result["risk_label"]
        if prob < 0.35:
            assert label == "Low", f"prob={prob} should be Low, got {label}"
        elif prob < 0.65:
            assert label == "Moderate", f"prob={prob} should be Moderate, got {label}"
        else:
            assert label == "High", f"prob={prob} should be High, got {label}"

    def test_latency_ms_positive(self, predictor, sample_patient):
        result = predictor.predict(sample_patient)
        assert result["latency_ms"] > 0

    def test_latency_under_500ms(self, predictor, sample_patient):
        """Single inference must complete in under 500ms (after model is loaded)."""
        _ = predictor.predict(sample_patient)  # warm up
        result = predictor.predict(sample_patient)
        assert result["latency_ms"] < 500, f"Inference too slow: {result['latency_ms']:.1f}ms"

    def test_deterministic_output(self, predictor, sample_patient):
        """Same input must produce identical output on repeated calls."""
        r1 = predictor.predict(sample_patient)
        r2 = predictor.predict(sample_patient)
        assert r1["prediction"] == r2["prediction"]
        assert r1["probability"] == r2["probability"]


# ---------------------------------------------------------------------------
# Batch prediction tests
# ---------------------------------------------------------------------------
class TestBatchPrediction:
    def test_batch_output_shape(self, predictor, sample_batch_df):
        result = predictor.predict_batch(sample_batch_df)
        assert len(result) == len(
            sample_batch_df
        ), f"Output rows ({len(result)}) != input rows ({len(sample_batch_df)})"

    def test_batch_has_required_columns(self, predictor, sample_batch_df):
        result = predictor.predict_batch(sample_batch_df)
        for col in ("prediction", "probability", "risk_label"):
            assert col in result.columns, f"Missing column: {col}"

    def test_batch_predictions_binary(self, predictor, sample_batch_df):
        result = predictor.predict_batch(sample_batch_df)
        assert set(result["prediction"].unique()) <= {
            0,
            1,
        }, f"Non-binary predictions: {result['prediction'].unique()}"

    def test_batch_probabilities_in_unit_interval(self, predictor, sample_batch_df):
        result = predictor.predict_batch(sample_batch_df)
        probs = result["probability"]
        assert (probs >= 0.0).all() and (
            probs <= 1.0
        ).all(), f"Probabilities out of [0,1]: min={probs.min()}, max={probs.max()}"

    def test_batch_risk_labels_valid(self, predictor, sample_batch_df):
        result = predictor.predict_batch(sample_batch_df)
        valid = {"Low", "Moderate", "High"}
        found = set(result["risk_label"].astype(str).unique()) - {"nan"}
        assert found <= valid, f"Unexpected risk labels: {found - valid}"

    def test_batch_preserves_original_columns(self, predictor, sample_batch_df):
        """predict_batch must not drop original input columns."""
        result = predictor.predict_batch(sample_batch_df)
        for col in sample_batch_df.columns:
            assert col in result.columns, f"Original column '{col}' was dropped"

    def test_single_row_batch(self, predictor, sample_patient):
        """Batch of 1 row should work identically to single predict()."""
        df = pd.DataFrame([sample_patient])
        result = predictor.predict_batch(df)
        single = predictor.predict(sample_patient)
        assert result["prediction"].iloc[0] == single["prediction"]
        assert abs(result["probability"].iloc[0] - single["probability"]) < 1e-4

    def test_large_batch(self, predictor, sample_patient):
        """Batch of 100 rows must complete without error."""
        df = pd.DataFrame([sample_patient] * 100)
        result = predictor.predict_batch(df)
        assert len(result) == 100


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------
class TestInputValidation:
    def test_valid_input_passes(self, sample_patient):
        cleaned = _validate_input(sample_patient)
        assert len(cleaned) == len(RAW_FEATURE_NAMES)

    def test_missing_required_feature_raises(self, sample_patient):
        bad = {k: v for k, v in sample_patient.items() if k != "age"}
        with pytest.raises(ValidationError, match="Missing required feature"):
            _validate_input(bad)

    def test_out_of_range_age_raises(self, sample_patient):
        bad = {**sample_patient, "age": 200}
        with pytest.raises(ValidationError, match="age"):
            _validate_input(bad)

    def test_out_of_range_chol_raises(self, sample_patient):
        bad = {**sample_patient, "chol": -10}
        with pytest.raises(ValidationError, match="chol"):
            _validate_input(bad)

    def test_invalid_sex_value_raises(self, sample_patient):
        bad = {**sample_patient, "sex": 5}
        with pytest.raises(ValidationError, match="sex"):
            _validate_input(bad)

    def test_null_required_feature_raises(self, sample_patient):
        bad = {**sample_patient, "age": None}
        with pytest.raises(ValidationError, match="age"):
            _validate_input(bad)

    def test_nullable_ca_accepts_none(self, sample_patient):
        """ca and thal are allowed to be None/NaN."""
        record = {**sample_patient, "ca": None}
        cleaned = _validate_input(record)
        assert np.isnan(cleaned["ca"])

    def test_nullable_thal_accepts_none(self, sample_patient):
        record = {**sample_patient, "thal": None}
        cleaned = _validate_input(record)
        assert np.isnan(cleaned["thal"])

    def test_string_numbers_coerced(self, sample_patient):
        """Numeric strings should be coerced, not rejected."""
        record = {**sample_patient, "age": "52", "chol": "212.0"}
        cleaned = _validate_input(record)
        assert cleaned["age"] == 52.0
        assert cleaned["chol"] == 212.0

    def test_predict_raises_on_bad_input(self, predictor):
        with pytest.raises(ValidationError):
            predictor.predict({"age": 52})  # missing 12 features


# ---------------------------------------------------------------------------
# ONNX tests (skipped if onnxruntime not installed)
# ---------------------------------------------------------------------------
onnxruntime = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")


class TestONNX:
    @pytest.fixture(scope="class")
    def onnx_path(self, models_dir, tmp_path_factory):
        """Export ONNX model to a temp file for testing."""
        from src.models.predict import export_to_onnx

        model_path = models_dir / "best_model.joblib"
        if not model_path.exists():
            pytest.skip("best_model.joblib not found — run train.py first.")

        out_path = tmp_path_factory.mktemp("onnx") / "best_model.onnx"
        return export_to_onnx(
            sklearn_model_path=model_path,
            onnx_output_path=out_path,
        )

    def test_onnx_file_created(self, onnx_path):
        assert onnx_path.exists(), "ONNX file was not created"
        assert onnx_path.stat().st_size > 0, "ONNX file is empty"

    def test_onnx_predictor_loads(self, models_dir):
        onnx_model_path = models_dir / "best_model.onnx"
        if not onnx_model_path.exists():
            pytest.skip("best_model.onnx not found — run --export first.")
        p = HeartDiseasePredictor(backend="onnx", model_path=onnx_model_path)
        assert not p.is_loaded

    def test_onnx_predictions_match_sklearn(self, predictor, models_dir, sample_batch_df):
        """ONNX and sklearn backends must agree on all predictions."""
        onnx_model_path = models_dir / "best_model.onnx"
        if not onnx_model_path.exists():
            pytest.skip("best_model.onnx not found — run --export first.")

        onnx_pred = HeartDiseasePredictor(backend="onnx", model_path=onnx_model_path)

        sk_result = predictor.predict_batch(sample_batch_df)
        on_result = onnx_pred.predict_batch(sample_batch_df)

        np.testing.assert_array_equal(
            sk_result["prediction"].values,
            on_result["prediction"].values,
            err_msg="ONNX and sklearn predictions disagree",
        )
        np.testing.assert_allclose(
            sk_result["probability"].values,
            on_result["probability"].values,
            atol=1e-4,
            err_msg="ONNX and sklearn probabilities differ by more than 1e-4",
        )

    def test_onnx_probability_in_unit_interval(self, models_dir, sample_batch_df):
        onnx_model_path = models_dir / "best_model.onnx"
        if not onnx_model_path.exists():
            pytest.skip("best_model.onnx not found.")

        onnx_pred = HeartDiseasePredictor(backend="onnx", model_path=onnx_model_path)
        result = onnx_pred.predict_batch(sample_batch_df)
        probs = result["probability"]
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
