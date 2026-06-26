"""
test_data.py — Unit tests for the preprocessing pipeline.


Run:
    pytest tests/test_data.py -v
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Expected schema after preprocessing
# ---------------------------------------------------------------------------
# 13 raw features → 20 processed features after OHE expansion:
#   continuous (5): age, trestbps, chol, thalach, oldpeak  → RobustScaler → 5
#   binary (3):     sex, fbs, exang                        → passthrough  → 3
#   nominal no-nan (3): cp, restecg, slope                 → OHE drop='first'
#                       cp(4 cats→3), restecg(3→2), slope(3→2) → 7
#   nominal with-nan (2): ca, thal                         → impute→OHE
#                         ca(4 cats→3), thal(3 cats→2)     → 5
# Total: 5 + 3 + 7 + 5 = 20
EXPECTED_N_FEATURES = 20

# Tolerance for 80/20 split size check (±5 rows handles rounding)
SPLIT_TOLERANCE = 5


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------
class TestShape:
    def test_X_train_columns(self, X_train):
        """Processed training set must have exactly EXPECTED_N_FEATURES columns."""
        assert X_train.shape[1] == EXPECTED_N_FEATURES, (
            f"Expected {EXPECTED_N_FEATURES} features, got {X_train.shape[1]}.\n"
            f"Columns: {X_train.columns.tolist()}"
        )

    def test_X_test_columns(self, X_test):
        """Processed test set must have the same number of columns as train."""
        assert (
            X_test.shape[1] == EXPECTED_N_FEATURES
        ), f"Expected {EXPECTED_N_FEATURES} features, got {X_test.shape[1]}"

    def test_train_test_same_columns(self, X_train, X_test):
        """Train and test must have identical column names in the same order."""
        assert (
            X_train.columns.tolist() == X_test.columns.tolist()
        ), "Column mismatch between X_train and X_test"

    def test_y_train_length_matches_X_train(self, X_train, y_train):
        assert len(X_train) == len(
            y_train
        ), f"X_train rows ({len(X_train)}) != y_train rows ({len(y_train)})"

    def test_y_test_length_matches_X_test(self, X_test, y_test):
        assert len(X_test) == len(
            y_test
        ), f"X_test rows ({len(X_test)}) != y_test rows ({len(y_test)})"

    def test_train_test_split_ratio(self, X_train, X_test):
        """Train should be ~80% of total data, test ~20%."""
        total = len(X_train) + len(X_test)
        expected_train = round(total * 0.80)
        expected_test = total - expected_train
        assert (
            abs(len(X_train) - expected_train) <= SPLIT_TOLERANCE
        ), f"Train size {len(X_train)} deviates from expected ~{expected_train}"
        assert (
            abs(len(X_test) - expected_test) <= SPLIT_TOLERANCE
        ), f"Test size {len(X_test)} deviates from expected ~{expected_test}"

    def test_minimum_rows(self, X_train, X_test):
        """Sanity check — at least 50 rows in each split."""
        assert len(X_train) >= 50, f"X_train too small: {len(X_train)} rows"
        assert len(X_test) >= 20, f"X_test too small: {len(X_test)} rows"


# ---------------------------------------------------------------------------
# Null / NaN tests
# ---------------------------------------------------------------------------
class TestNulls:
    def test_X_train_no_nulls(self, X_train):
        null_counts = X_train.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert (
            cols_with_nulls.empty
        ), f"X_train contains nulls after preprocessing:\n{cols_with_nulls}"

    def test_X_test_no_nulls(self, X_test):
        null_counts = X_test.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert (
            cols_with_nulls.empty
        ), f"X_test contains nulls after preprocessing:\n{cols_with_nulls}"

    def test_y_train_no_nulls(self, y_train):
        assert y_train.isnull().sum() == 0, "y_train contains null values"

    def test_y_test_no_nulls(self, y_test):
        assert y_test.isnull().sum() == 0, "y_test contains null values"


# ---------------------------------------------------------------------------
# Dtype tests
# ---------------------------------------------------------------------------
class TestDtypes:
    def test_X_train_numeric(self, X_train):
        """All processed feature columns must be numeric."""
        non_numeric = [
            col for col in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[col])
        ]
        assert not non_numeric, f"Non-numeric columns in X_train: {non_numeric}"

    def test_X_test_numeric(self, X_test):
        non_numeric = [
            col for col in X_test.columns if not pd.api.types.is_numeric_dtype(X_test[col])
        ]
        assert not non_numeric, f"Non-numeric columns in X_test: {non_numeric}"

    def test_y_train_integer(self, y_train):
        """Target must be integer dtype (0 or 1)."""
        assert pd.api.types.is_integer_dtype(y_train) or pd.api.types.is_float_dtype(
            y_train
        ), f"y_train dtype unexpected: {y_train.dtype}"

    def test_y_test_integer(self, y_test):
        assert pd.api.types.is_integer_dtype(y_test) or pd.api.types.is_float_dtype(
            y_test
        ), f"y_test dtype unexpected: {y_test.dtype}"


# ---------------------------------------------------------------------------
# Target binary tests
# ---------------------------------------------------------------------------
class TestTarget:
    def test_y_train_binary(self, y_train):
        """Target must contain only 0 and 1."""
        unique = set(y_train.unique())
        assert unique <= {0, 1}, f"y_train contains non-binary values: {unique}"

    def test_y_test_binary(self, y_test):
        unique = set(y_test.unique())
        assert unique <= {0, 1}, f"y_test contains non-binary values: {unique}"

    def test_y_train_both_classes_present(self, y_train):
        """Both classes must be present in training set."""
        unique = set(y_train.unique())
        assert 0 in unique and 1 in unique, f"y_train is missing a class: found {unique}"

    def test_y_test_both_classes_present(self, y_test):
        unique = set(y_test.unique())
        assert 0 in unique and 1 in unique, f"y_test is missing a class: found {unique}"

    def test_class_balance_preserved(self, y_train, y_test):
        """
        Stratified split should preserve class ratio within ±5% between splits.
        """
        train_ratio = y_train.mean()  # proportion of class 1
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05, (
            f"Class ratio drift between splits: " f"train={train_ratio:.3f}, test={test_ratio:.3f}"
        )


# ---------------------------------------------------------------------------
# Feature value range tests (sanity bounds post-scaling)
# ---------------------------------------------------------------------------
class TestFeatureValues:
    def test_no_infinite_values_train(self, X_train):
        assert not np.isinf(X_train.values).any(), "X_train contains infinite values"

    def test_no_infinite_values_test(self, X_test):
        assert not np.isinf(X_test.values).any(), "X_test contains infinite values"

    def test_ohe_columns_binary(self, X_train):
        """
        One-hot encoded columns must contain only 0.0 and 1.0.
        OHE columns are identifiable by the '__' separator in their name.
        """
        ohe_cols = [c for c in X_train.columns if "__" in c]
        if not ohe_cols:
            pytest.skip("No OHE columns found (column naming may differ)")
        for col in ohe_cols:
            unique_vals = set(X_train[col].unique())
            assert unique_vals <= {
                0.0,
                1.0,
                0,
                1,
            }, f"OHE column '{col}' has unexpected values: {unique_vals}"


# ---------------------------------------------------------------------------
# Preprocessor Pipeline test
# ---------------------------------------------------------------------------
class TestPreprocessorPipeline:
    def test_pipeline_contains_preprocessor(self, models_dir):
        """
        best_model.joblib must be a sklearn Pipeline with a 'preprocessor' step.
        This is the core Step 4.1 requirement.
        """
        import joblib
        from sklearn.pipeline import Pipeline

        model_path = models_dir / "best_model.joblib"
        if not model_path.exists():
            pytest.skip("best_model.joblib not found — run train.py first.")

        pipeline = joblib.load(model_path)

        assert isinstance(
            pipeline, Pipeline
        ), f"best_model.joblib is {type(pipeline).__name__}, expected sklearn Pipeline"
        assert (
            "preprocessor" in pipeline.named_steps
        ), f"Pipeline missing 'preprocessor' step. Steps: {list(pipeline.named_steps)}"
        assert (
            "classifier" in pipeline.named_steps
        ), f"Pipeline missing 'classifier' step. Steps: {list(pipeline.named_steps)}"

    def test_pipeline_accepts_raw_features(self, models_dir, sample_patient):
        """
        The saved Pipeline must accept raw 13-feature input and produce a prediction
        without requiring any external preprocessor file.
        """
        import joblib
        import sys

        sys.path.insert(0, str(models_dir.parent))
        from src.models.predict import RAW_FEATURE_NAMES

        model_path = models_dir / "best_model.joblib"
        if not model_path.exists():
            pytest.skip("best_model.joblib not found — run train.py first.")

        pipeline = joblib.load(model_path)
        df = pd.DataFrame([sample_patient], columns=RAW_FEATURE_NAMES)
        pred = pipeline.predict(df)

        assert len(pred) == 1
        assert pred[0] in (0, 1)
