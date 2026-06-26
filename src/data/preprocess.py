"""

Cleans and transforms the raw Heart Disease UCI dataset into a
model-ready feature matrix. Produces two artifacts:

    data/processed/X_train.csv   — training features
    data/processed/X_test.csv    — test features
    data/processed/y_train.csv   — training labels
    data/processed/y_test.csv    — test labels
    data/processed/preprocessor.joblib — fitted sklearn ColumnTransformer
                                         (used verbatim at inference time)

Design principles
-----------------
* All transformations are wrapped in a single sklearn ColumnTransformer so
  the same object can be serialised and reused inside training and the API —
  no silent train/test leakage, no duplicated logic.
* Categorical columns with medical meaning (cp, thal, restecg) are
  one-hot encoded rather than ordinally encoded, because their integer
  codes do NOT imply a linear ordering.
* Continuous columns are scaled with RobustScaler (median + IQR) instead
  of StandardScaler because cholesterol and blood pressure have meaningful
  clinical outliers that should be preserved in proportion.
* Missing values (only 'ca' and 'thal' in the UCI data) are imputed with
  the most-frequent value before encoding — appropriate for near-nominal
  columns with very few missings (~1-2%).

"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# Continuous features: scaled with RobustScaler
CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

BINARY_COLS = ["sex", "fbs", "exang"]


NOMINAL_COLS = ["cp", "restecg", "slope"]


NOMINAL_WITH_NAN_COLS = ["ca", "thal"]

# Derived: drop before modelling (source tag from download script)
DROP_COLS = ["source"]

ALL_FEATURE_COLS = CONTINUOUS_COLS + BINARY_COLS + NOMINAL_COLS + NOMINAL_WITH_NAN_COLS

TARGET_COL = "target"


# ---------------------------------------------------------------------------
# Step 1 — Load raw data
# ---------------------------------------------------------------------------


def load_raw(path: Path) -> pd.DataFrame:
    """Load CSV produced by download_data.py."""
    log.info(f"Loading raw data from: {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}\n" "Run: python src/data/download_data.py"
        )
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Basic data cleaning
# ---------------------------------------------------------------------------


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """


    1. Drop metadata columns (source tag).
    2. Binarise target: UCI uses 0=no disease, 1-4=disease severity.
       We collapse to 0/1.
    3. Cast 'ca' and 'thal' to nullable int (they arrive as float due to NaN).
    4. Validate feature value ranges and warn on anomalies.
    5. Drop exact duplicate rows.
    """
    log.info("--- Step 2: Cleaning ---")
    original_len = len(df)

    # 2a. Drop metadata columns not used in modelling
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        log.info(f"Dropped metadata columns: {cols_to_drop}")

    # 2b. Binarise target (0=healthy, 1=disease)
    if df[TARGET_COL].max() > 1:
        pre = df[TARGET_COL].value_counts().sort_index().to_dict()
        df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)
        post = df[TARGET_COL].value_counts().sort_index().to_dict()
        log.info(f"Target binarised: {pre} → {post}")
    else:
        log.info("Target already binary — skipping binarisation.")

    # 2c. Type coercion: ca and thal contain meaningful NaNs — keep as float
    #     but ensure they only hold their valid domain values
    for col, valid in {"ca": {0, 1, 2, 3}, "thal": {3, 6, 7}}.items():
        if col not in df.columns:
            continue
        invalid_mask = df[col].notna() & ~df[col].isin(valid)
        if invalid_mask.any():
            n_invalid = invalid_mask.sum()
            log.warning(
                f"Column '{col}': {n_invalid} out-of-domain values "
                f"({df.loc[invalid_mask, col].unique()}) → set to NaN"
            )
            df.loc[invalid_mask, col] = np.nan

    # 2d. Range validation — warn but do NOT silently drop clinical outliers
    range_checks = {
        "age": (1, 120),
        "trestbps": (50, 250),
        "chol": (50, 700),
        "thalach": (40, 250),
        "oldpeak": (0, 10),
    }
    for col, (lo, hi) in range_checks.items():
        if col not in df.columns:
            continue
        out = df[(df[col] < lo) | (df[col] > hi)]
        if len(out) > 0:
            log.warning(
                f"Column '{col}': {len(out)} values outside [{lo}, {hi}] "
                f"— kept as-is (RobustScaler handles outliers)"
            )

    # 2e. Drop exact duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        df = df.drop_duplicates()
        log.warning(f"Dropped {n_dupes} exact duplicate rows.")

    log.info(
        f"Cleaning complete: {original_len} → {len(df)} rows " f"({original_len - len(df)} removed)"
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3 — Split features / target
# ---------------------------------------------------------------------------


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separates feature matrix X from target series y.
    Only columns in ALL_FEATURE_COLS are kept — extra columns are silently
    ignored, making this robust to datasets with different sub-source columns.
    """
    missing = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required feature columns: {missing}\n"
            "Check that download_data.py produced a complete dataset."
        )

    X = df[ALL_FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    log.info(
        f"Features: {X.shape[1]} columns, {X.shape[0]} rows  |  "
        f"Target: {y.value_counts().to_dict()}"
    )
    return X, y


# ---------------------------------------------------------------------------
# Step 4 — Build preprocessing pipeline
# ---------------------------------------------------------------------------


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs the sklearn ColumnTransformer.

    Transformer A — continuous features:
        RobustScaler  (median + IQR normalisation, outlier-resistant)
        No imputation needed: continuous cols have no missing values in UCI.

    Transformer B — binary features:
        Pass-through (already 0/1, no transformation)

    Transformer C — nominal features (no NaN):
        OneHotEncoder with drop='first' (avoids multicollinearity)
        handle_unknown='ignore' (safe for unseen categories at inference)

    Transformer D — nominal features WITH NaN (ca, thal):
        SimpleImputer (most_frequent) → OneHotEncoder
        Chained in a sub-Pipeline so imputation runs first.

    remainder='drop' ensures any unexpected columns are excluded silently,
    preventing silent data leakage if new columns are added upstream.
    """

    # A: Continuous
    continuous_transformer = Pipeline(
        steps=[
            ("scaler", RobustScaler()),
        ]
    )

    # B: Binary — passthrough (sklearn will leave these as-is)
    # Handled via remainder or explicit passthrough below

    # C: Nominal without NaN
    nominal_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    drop="first",  # k-1 encoding, avoids dummy trap
                    handle_unknown="ignore",
                    sparse_output=False,  # return dense array (easier to debug)
                ),
            ),
        ]
    )

    # D: Nominal with NaN — impute then encode
    nominal_nan_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous_transformer, CONTINUOUS_COLS),
            ("binary", "passthrough", BINARY_COLS),
            ("nominal", nominal_transformer, NOMINAL_COLS),
            ("nominal_nan", nominal_nan_transformer, NOMINAL_WITH_NAN_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,  # clean column names (no prefix)
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Returns human-readable output feature names after fitting.
    Used for logging and feature importance plots in train.py.
    """
    return list(preprocessor.get_feature_names_out())


# ---------------------------------------------------------------------------
# Step 5 — Train/test split
# ---------------------------------------------------------------------------


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified split to preserve class balance in both partitions.
    Stratification is non-negotiable for a medical binary classifier.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,  # preserve 0/1 ratio in both splits
    )
    log.info(
        f"Train: {len(X_train)} rows  |  "
        f"Test:  {len(X_test)} rows  |  "
        f"Test size: {test_size:.0%}  |  "
        f"Seed: {seed}"
    )
    log.info(f"Train class balance: " f"{y_train.value_counts(normalize=True).round(3).to_dict()}")
    log.info(f"Test class balance:  " f"{y_test.value_counts(normalize=True).round(3).to_dict()}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 6 — Fit & transform
# ---------------------------------------------------------------------------


def fit_transform(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fits the ColumnTransformer on X_train only, then applies to both.
    Returns DataFrames (not arrays) with named columns for interpretability.

    CRITICAL: preprocessor.fit() is ONLY called on X_train.
    Calling fit_transform on the full dataset before splitting would leak
    test statistics (mean, median, mode) into training — a common mistake.
    """
    log.info("Fitting preprocessor on training data only...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, dtype=float)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, dtype=float)

    log.info(
        f"Processed feature matrix: "
        f"{X_train_df.shape[1]} features "
        f"(from {len(ALL_FEATURE_COLS)} raw columns)"
    )
    log.info(f"Output features: {feature_names}")
    return X_train_df, X_test_df


# ---------------------------------------------------------------------------
# Step 7 — Save outputs
# ---------------------------------------------------------------------------


def save_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    output_dir: Path,
) -> None:
    """Saves all four CSV splits and the fitted preprocessor."""
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False, header=True)
    y_test.to_csv(output_dir / "y_test.csv", index=False, header=True)

    preprocessor_path = output_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    log.info(f"Saved X_train.csv  : {X_train.shape}")
    log.info(f"Saved X_test.csv   : {X_test.shape}")
    log.info(f"Saved y_train.csv  : {y_train.shape}")
    log.info(f"Saved y_test.csv   : {y_test.shape}")
    log.info(f"Saved preprocessor : {preprocessor_path}")


def run_preprocessing(
    input_path: Path,
    output_dir: Path,
    test_size: float = 0.20,
    seed: int = 42,
) -> dict:
    """
    Runs the full preprocessing pipeline end-to-end.

    Returns a summary dict so train.py can log these values to MLflow
    without re-reading the files.

    Parameters
    ----------
    input_path  : Path to raw CSV (output of download_data.py)
    output_dir  : Directory to write processed splits + preprocessor
    test_size   : Fraction of data reserved for testing (default 0.20)
    seed        : Random seed for reproducibility (default 42)

    Returns
    -------
    dict with keys: n_train, n_test, n_features, feature_names,
                    class_balance_train, class_balance_test
    """
    log.info("=" * 60)
    log.info("Preprocessing Pipeline — Heart Disease UCI Dataset")
    log.info("=" * 60)

    df = load_raw(input_path)
    df = clean(df)
    X, y = split_features_target(df)
    X_train_raw, X_test_raw, y_train, y_test = split_train_test(
        X, y, test_size=test_size, seed=seed
    )
    preprocessor = build_preprocessor()
    X_train, X_test = fit_transform(preprocessor, X_train_raw, X_test_raw)
    save_outputs(X_train, X_test, y_train, y_test, preprocessor, output_dir)

    summary = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_raw_features": len(ALL_FEATURE_COLS),
        "n_processed_features": X_train.shape[1],
        "feature_names": list(X_train.columns),
        "class_balance_train": (y_train.value_counts(normalize=True).round(4).to_dict()),
        "class_balance_test": (y_test.value_counts(normalize=True).round(4).to_dict()),
    }

    log.info("=" * 60)
    log.info("Preprocessing complete.")
    log.info(f"  Train rows      : {summary['n_train']}")
    log.info(f"  Test rows       : {summary['n_test']}")
    log.info(f"  Raw features    : {summary['n_raw_features']}")
    log.info(f"  Output features : {summary['n_processed_features']}")
    log.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess the Heart Disease UCI dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/heart_disease_raw.csv",
        help="Path to raw CSV from download_data.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Directory to write processed splits (default: data/processed)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Test split fraction, e.g. 0.20 = 20%% (default: 0.20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    run_preprocessing(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
