"""
download_data.py
----------------
Downloads the Heart Disease UCI Dataset using one of three methods:
    1. ucimlrepo  - Official UCI Python package (preferred)
    2. Direct URL - Raw CSV from UCI archive (fallback)
    3. Kaggle API - Via kaggle CLI (fallback if UCI is unreachable)

Usage:
    python src/data/download_data.py                    # auto (tries methods 1→2→3)
    python src/data/download_data.py --method uci       # force ucimlrepo
    python src/data/download_data.py --method direct    # force direct URL
    python src/data/download_data.py --method kaggle    # force Kaggle API
    python src/data/download_data.py --output data/raw  # custom output path

Output:
    data/raw/heart_disease_raw.csv   — combined dataset (all 4 sources merged)
    data/raw/download_report.json    — metadata about the download
"""

import argparse
import json
import logging
import sys
import urllib.request
import ssl
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# UCI Heart Disease dataset: 4 sub-datasets (Cleveland is the most used,
# but we combine all 4 to create a richer, more original dataset)
_UCI_BASE = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
)
UCI_DIRECT_URLS = {
    "cleveland":  f"{_UCI_BASE}/processed.cleveland.data",
    "hungarian":  f"{_UCI_BASE}/processed.hungarian.data",
    "switzerland": f"{_UCI_BASE}/processed.switzerland.data",
    "va":         f"{_UCI_BASE}/processed.va.data",
}

# Column names as defined in the UCI dataset documentation
COLUMN_NAMES = [
    "age",       # age in years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type (1-4)
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl (1=true, 0=false)
    "restecg",   # resting ECG results (0-2)
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise induced angina (1=yes, 0=no)
    "oldpeak",   # ST depression induced by exercise vs rest
    "slope",     # slope of peak exercise ST segment (1-3)
    "ca",        # major vessels coloured by fluoroscopy (0-3)
    "thal",      # thalassemia (3=normal, 6=fixed, 7=reversible)
    "target",    # diagnosis (0=no disease, 1-4=disease; binarised)
]

KAGGLE_DATASET = "ronitf/heart-disease-uci"   # fallback Kaggle dataset slug
OUTPUT_FILENAME = "heart_disease_raw.csv"
REPORT_FILENAME = "download_report.json"


# ---------------------------------------------------------------------------
# Download Methods
# ---------------------------------------------------------------------------

def download_via_ucimlrepo(output_dir: Path) -> pd.DataFrame:
    """
    Method 1: Uses the official `ucimlrepo` package.
    Fetches dataset id=45 (Heart Disease).
    Returns a combined features+target DataFrame.
    """
    log.info("Method 1: Attempting download via ucimlrepo package...")

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError(
            "ucimlrepo is not installed. Run: pip install ucimlrepo"
        )

    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    # ucimlrepo may name the target column differently — normalise it
    target_col = y.columns[0]
    if target_col != "target":
        df.rename(columns={target_col: "target"}, inplace=True)

    log.info(f"ucimlrepo download successful — shape: {df.shape}")
    return df


def download_via_direct_url(output_dir: Path) -> pd.DataFrame:
    """
    Method 2: Downloads all 4 processed sub-datasets directly from the
    UCI archive and merges them.
    Uses an unverified SSL context as a fallback for corporate/restricted networks.
    """
    log.info("Method 2: Attempting direct URL download from UCI archive...")

    frames = []
    ssl_ctx = ssl.create_default_context()

    for source_name, url in UCI_DIRECT_URLS.items():
        log.info(f"  Fetching {source_name} dataset from: {url}")

        try:
            with urllib.request.urlopen(url, context=ssl_ctx, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
        except ssl.SSLError:
            log.warning("  SSL verification failed — retrying without certificate check...")
            ssl_ctx_unverified = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=ssl_ctx_unverified, timeout=30) as resp:
                raw = resp.read().decode("utf-8")

        from io import StringIO
        df_part = pd.read_csv(
            StringIO(raw),
            header=None,
            names=COLUMN_NAMES,
            na_values="?",      # UCI uses '?' for missing values
        )
        df_part["source"] = source_name  # track origin for analysis
        frames.append(df_part)
        log.info(f"  {source_name}: {len(df_part)} rows loaded")

    df = pd.concat(frames, ignore_index=True)
    log.info(f"Direct URL download successful — combined shape: {df.shape}")
    return df


def download_via_kaggle(output_dir: Path) -> pd.DataFrame:
    """
    Method 3: Downloads via the Kaggle API.
    Requires ~/.kaggle/kaggle.json with your API credentials.
    Install: pip install kaggle
    Setup:   https://www.kaggle.com/docs/api#authentication
    """
    log.info("Method 3: Attempting download via Kaggle API...")

    try:
        import kaggle  # noqa: F401 — triggers credential check on import
    except ImportError:
        raise ImportError(
            "kaggle package not installed. Run: pip install kaggle"
        )
    except OSError as e:
        raise OSError(
            f"Kaggle credentials not found: {e}\n"
            "Place your kaggle.json at ~/.kaggle/kaggle.json\n"
            "Get it from: https://www.kaggle.com/settings → API → Create New Token"
        )

    import subprocess
    kaggle_dir = output_dir / "kaggle_tmp"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(kaggle_dir),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle CLI failed:\n{result.stderr}"
        )

    # Find the downloaded CSV
    csv_files = list(kaggle_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV found in {kaggle_dir} after Kaggle download."
        )

    df = pd.read_csv(csv_files[0])
    log.info(f"Kaggle download successful — shape: {df.shape}")

    # Clean up temp dir
    import shutil
    shutil.rmtree(kaggle_dir)

    return df


# ---------------------------------------------------------------------------
# Post-Download Processing
# ---------------------------------------------------------------------------

def binarise_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    The UCI target column has values 0–4 where:
        0 = no heart disease
        1, 2, 3, 4 = presence of heart disease (varying severity)

    Standard practice is to binarise: 0 = no disease, 1 = disease.
    Kaggle version already has binary targets, so this is a no-op if max == 1.
    """
    if df["target"].max() > 1:
        original_counts = df["target"].value_counts().to_dict()
        df["target"] = (df["target"] > 0).astype(int)
        log.info(
            f"Target binarised: original distribution {original_counts} "
            f"→ new: {df['target'].value_counts().to_dict()}"
        )
    else:
        log.info("Target already binary — no binarisation needed.")
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Runs basic integrity checks on the downloaded dataset.
    Returns a validation report dictionary.
    """
    log.info("Running dataset validation...")

    required_cols = set(COLUMN_NAMES)
    # 'source' column may or may not be present depending on method
    present_cols = set(df.columns) - {"source"}

    missing_cols = required_cols - present_cols
    extra_cols = present_cols - required_cols

    report = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_required_columns": list(missing_cols),
        "unexpected_columns": list(extra_cols),
        "missing_value_counts": df.isnull().sum().to_dict(),
        "total_missing_values": int(df.isnull().sum().sum()),
        "target_distribution": df["target"].value_counts().to_dict(),
        "class_balance_pct": (
            df["target"].value_counts(normalize=True) * 100
        ).round(2).to_dict(),
        "feature_dtypes": df.dtypes.astype(str).to_dict(),
    }

    if missing_cols:
        log.warning(f"Missing required columns: {missing_cols}")
    else:
        log.info("All required columns present.")

    log.info(f"Total rows: {report['row_count']}")
    log.info(f"Total missing values: {report['total_missing_values']}")
    log.info(f"Target distribution: {report['target_distribution']}")

    return report


def save_dataset(df: pd.DataFrame, output_dir: Path) -> Path:
    """Saves the dataset CSV to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    df.to_csv(output_path, index=False)
    log.info(f"Dataset saved to: {output_path}")
    return output_path


def save_report(report: dict, output_dir: Path, method_used: str) -> Path:
    """Saves a JSON download report for traceability."""
    report["download_timestamp"] = datetime.utcnow().isoformat() + "Z"
    report["method_used"] = method_used
    report["output_file"] = OUTPUT_FILENAME

    report_path = output_dir / REPORT_FILENAME
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Download report saved to: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def download_dataset(method: str, output_dir: Path) -> tuple[pd.DataFrame, str]:
    """
    Tries download methods in order until one succeeds.
    Returns (DataFrame, method_name_that_worked).
    """
    methods = {
        "uci":     download_via_ucimlrepo,
        "direct":  download_via_direct_url,
        "kaggle":  download_via_kaggle,
    }

    if method != "auto":
        # Force a specific method
        if method not in methods:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {list(methods.keys())} or 'auto'"
            )
        df = methods[method](output_dir)
        return df, method

    # Auto: try in priority order
    priority = ["uci", "direct", "kaggle"]
    last_error = None

    for m in priority:
        try:
            df = methods[m](output_dir)
            return df, m
        except Exception as e:
            log.warning(f"Method '{m}' failed: {e}")
            last_error = e

    raise RuntimeError(
        f"All download methods failed. Last error: {last_error}\n"
        "Check your internet connection or set up Kaggle credentials."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download the Heart Disease UCI Dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--method",
        choices=["auto", "uci", "direct", "kaggle"],
        default="auto",
        help="Download method (default: auto — tries uci → direct → kaggle)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory for the raw dataset (default: data/raw)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    log.info("=" * 60)
    log.info("Heart Disease UCI Dataset — Download Script")
    log.info("=" * 60)
    log.info(f"Method: {args.method}")
    log.info(f"Output: {output_dir.resolve()}")

    # Download
    df, method_used = download_dataset(args.method, output_dir)

    # Binarise target (UCI has 0–4, we want 0/1)
    df = binarise_target(df)

    # Validate
    report = validate_dataset(df)

    # Save
    save_dataset(df, output_dir)
    save_report(report, output_dir, method_used)

    log.info("=" * 60)
    log.info("Download complete.")
    log.info(f"  Rows:            {report['row_count']}")
    log.info(f"  Missing values:  {report['total_missing_values']}")
    log.info(f"  Class balance:   {report['class_balance_pct']}")
    log.info(f"  Method used:     {method_used}")
    log.info("=" * 60)

    # Exit with error code if critical columns are missing
    if report["missing_required_columns"]:
        log.error(
            f"Dataset is missing required columns: {report['missing_required_columns']}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()