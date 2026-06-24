"""
train.py — Model training, hyperparameter tuning, and MLflow experiment tracking.

Trains three classifiers on the preprocessed Heart Disease dataset:
  - Logistic Regression  (interpretable baseline)
  - Random Forest        (ensemble, non-linear)
  - XGBoost              (gradient boosting, typically best)

Usage:
    python -m src.models.train              # full run with tuning
    python -m src.models.train --no-tune   # skip GridSearchCV (CI/fast mode)

Importable:
    from src.models.train import run_training
    results = run_training(tune=False, cv_folds=2)  # for CI smoke tests
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
# MLflow tracking URI
# Uses SQLite to stay compatible with both MLflow 2.x (requirements.txt pin)
# and MLflow 3.x (what the sandbox may have installed).
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "heart-disease-classification"

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def _build_model_configs(tune: bool) -> list[dict[str, Any]]:
    """
    Return a list of model configs.
    Each config has: name, estimator, param_grid (empty when tune=False).
    """
    configs = [
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            ),
            "param_grid": (
                {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "penalty": ["l2"],
                }
                if tune
                else {}
            ),
        },
        {
            "name": "RandomForest",
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
            "param_grid": (
                {
                    "n_estimators": [100, 300],
                    "max_depth": [None, 6],
                    "min_samples_leaf": [1, 4],
                }
                if tune
                else {}
            ),
        },
        {
            "name": "XGBoost",
            "estimator": XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            ),
            "param_grid": (
                {
                    "n_estimators": [100, 300],
                    "max_depth": [3, 6],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.8, 1.0],
                }
                if tune
                else {}
            ),
        },
    ]
    return configs


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _compute_test_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "test_accuracy": round(accuracy_score(y_true, y_pred), 4),
        "test_roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "test_f1": round(f1_score(y_true, y_pred), 4),
        "test_precision": round(precision_score(y_true, y_pred), 4),
        "test_recall": round(recall_score(y_true, y_pred), 4),
    }


def _plot_roc_curve(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str,
    color: str,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {auc:.3f})")


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path,
) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        title=f"Confusion Matrix — {model_name}",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No Disease (0)", "Disease (1)"],
        yticklabels=["No Disease (0)", "Disease (1)"],
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return save_path


def _save_feature_importance(
    estimator: Any,
    feature_names: list[str],
    model_name: str,
    save_path: Path,
) -> Path | None:
    """Save feature importance plot if the estimator supports it."""
    importances = None

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])

    if importances is None:
        return None

    idx = np.argsort(importances)[::-1][:20]  # top 20
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(idx)), importances[idx])
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Feature Importances — {model_name}")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Per-model training loop
# ---------------------------------------------------------------------------
def _train_single_model(
    config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    cv_folds: int,
    experiment_id: str,
) -> dict:
    """
    Trains one model (with optional GridSearchCV), logs to MLflow, returns summary dict.
    """
    model_name = config["name"]
    estimator = config["estimator"]
    param_grid = config["param_grid"]
    tune = bool(param_grid)

    log.info("Training: %s (tune=%s)", model_name, tune)
    t0 = time.time()

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    with mlflow.start_run(run_name=model_name, experiment_id=experiment_id) as run:
        run_id = run.info.run_id

        # ------------------------------------------------------------------
        # Hyperparameter tuning
        # ------------------------------------------------------------------
        if tune:
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
                return_train_score=False,
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            best_params = search.best_params_
            cv_mean_auc = search.best_score_
            cv_std_auc = search.cv_results_["std_test_score"][search.best_index_]
        else:
            estimator.fit(X_train, y_train)
            best_estimator = estimator
            best_params = {}
            cv_mean_auc = 0.0
            cv_std_auc = 0.0

        elapsed = round(time.time() - t0, 1)

        # ------------------------------------------------------------------
        # Test set evaluation
        # ------------------------------------------------------------------
        y_pred = best_estimator.predict(X_test)
        y_prob = best_estimator.predict_proba(X_test)[:, 1]
        test_metrics = _compute_test_metrics(y_test, y_pred, y_prob)

        # ------------------------------------------------------------------
        # MLflow logging — params
        # ------------------------------------------------------------------
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("tuned", tune)
        mlflow.log_param("cv_folds", cv_folds)
        for k, v in best_params.items():
            mlflow.log_param(k, v)

        # ------------------------------------------------------------------
        # MLflow logging — metrics
        # ------------------------------------------------------------------
        if tune:
            mlflow.log_metric("cv_mean_roc_auc", round(cv_mean_auc, 4))
            mlflow.log_metric("cv_std_roc_auc", round(cv_std_auc, 4))

        for metric_name, metric_val in test_metrics.items():
            mlflow.log_metric(metric_name, metric_val)
        mlflow.log_metric("training_time_seconds", elapsed)

        # ------------------------------------------------------------------
        # MLflow logging — artifacts
        # ------------------------------------------------------------------
        artifact_dir = MODELS_DIR / "artifacts" / model_name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        cm_path = _save_confusion_matrix(
            y_test, y_pred, model_name, artifact_dir / "confusion_matrix.png"
        )
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # Feature importance
        fi_path = _save_feature_importance(
            best_estimator, feature_names, model_name, artifact_dir / "feature_importance.png"
        )
        if fi_path:
            mlflow.log_artifact(str(fi_path), artifact_path="plots")

        # Classification report as JSON
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = artifact_dir / "classification_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        # Model artifact with input example
        input_example = pd.DataFrame(X_train[:5], columns=feature_names)
        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            artifact_path="model",
            input_example=input_example,
        )

        log.info(
            "%s done in %.1fs | AUC=%.4f | F1=%.4f",
            model_name,
            elapsed,
            test_metrics["test_roc_auc"],
            test_metrics["test_f1"],
        )

    return {
        "model_name": model_name,
        "run_id": run_id,
        "best_params": best_params,
        "cv_mean_roc_auc": round(cv_mean_auc, 4),
        "cv_std_roc_auc": round(cv_std_auc, 4),
        "test_metrics": test_metrics,
        "best_estimator": best_estimator,
        "y_prob": y_prob,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# ROC comparison chart (all three models on one axes)
# ---------------------------------------------------------------------------
_ROC_COLORS = {"LogisticRegression": "#1f77b4", "RandomForest": "#2ca02c", "XGBoost": "#d62728"}


def _save_roc_comparison(results: list[dict], y_test: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        _plot_roc_curve(
            ax,
            y_test,
            r["y_prob"],
            label=r["model_name"],
            color=_ROC_COLORS.get(r["model_name"], "grey"),
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curves — All Models",
        xlim=[0, 1],
        ylim=[0, 1.02],
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    log.info("ROC comparison saved → %s", save_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_training(tune: bool = True, cv_folds: int = 5) -> dict:
    """
    Full training pipeline.

    Parameters
    ----------
    tune : bool
        Run GridSearchCV if True. Set False for CI smoke tests.
    cv_folds : int
        Number of CV folds. Use 2 for CI, 5 for production.

    Returns
    -------
    dict
        Summary with per-model metrics and path to best model.
    """
    # ------------------------------------------------------------------
    # Load preprocessed data
    # ------------------------------------------------------------------
    log.info("Loading preprocessed splits from %s", PROCESSED_DIR)
    try:
        X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
        X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
        y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
        y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    except FileNotFoundError as exc:
        log.error("Preprocessed data not found — run preprocess.py first.\n%s", exc)
        sys.exit(1)

    feature_names = X_train.columns.tolist()
    log.info(
        "Data loaded | train=%d  test=%d  features=%d",
        len(X_train),
        len(X_test),
        len(feature_names),
    )

    # ------------------------------------------------------------------
    # MLflow experiment setup
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
    log.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)
    log.info("Experiment: %s (id=%s)", EXPERIMENT_NAME, experiment_id)

    # ------------------------------------------------------------------
    # Train all models
    # ------------------------------------------------------------------
    configs = _build_model_configs(tune=tune)
    all_results: list[dict] = []

    for config in configs:
        result = _train_single_model(
            config=config,
            X_train=X_train.values,
            y_train=y_train.values,
            X_test=X_test.values,
            y_test=y_test.values,
            feature_names=feature_names,
            cv_folds=cv_folds,
            experiment_id=experiment_id,
        )
        all_results.append(result)

    # ------------------------------------------------------------------
    # Combined ROC comparison chart
    # ------------------------------------------------------------------
    roc_path = MODELS_DIR / "roc_comparison.png"
    _save_roc_comparison(all_results, y_test.values, roc_path)

    # ------------------------------------------------------------------
    # Pick best model by test ROC-AUC
    # ------------------------------------------------------------------
    best = max(all_results, key=lambda r: r["test_metrics"]["test_roc_auc"])
    log.info(
        "Best model: %s | AUC=%.4f",
        best["model_name"],
        best["test_metrics"]["test_roc_auc"],
    )

    # Save best model artifact
    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best["best_estimator"], best_model_path)
    log.info("Best model saved → %s", best_model_path)

    # Save best model summary JSON
    summary = {
        "best_model": best["model_name"],
        "mlflow_run_id": best["run_id"],
        "best_params": best["best_params"],
        "cv_mean_roc_auc": best["cv_mean_roc_auc"],
        "cv_std_roc_auc": best["cv_std_roc_auc"],
        "test_metrics": best["test_metrics"],
        "all_models": [
            {
                "model_name": r["model_name"],
                "test_metrics": r["test_metrics"],
                "elapsed_seconds": r["elapsed"],
            }
            for r in all_results
        ],
    }
    summary_path = MODELS_DIR / "best_model_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Summary saved → %s", summary_path)

    # ------------------------------------------------------------------
    # Print leaderboard to stdout
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"{'MODEL':<22} {'AUC':>8} {'F1':>8} {'PREC':>8} {'REC':>8}")
    print("-" * 65)
    for r in sorted(all_results, key=lambda x: x["test_metrics"]["test_roc_auc"], reverse=True):
        m = r["test_metrics"]
        marker = " ← best" if r["model_name"] == best["model_name"] else ""
        print(
            f"{r['model_name']:<22} {m['test_roc_auc']:>8.4f} "
            f"{m['test_f1']:>8.4f} {m['test_precision']:>8.4f} "
            f"{m['test_recall']:>8.4f}{marker}"
        )
    print("=" * 65 + "\n")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train heart disease classifiers.")
    parser.add_argument(
        "--no-tune",
        dest="tune",
        action="store_false",
        help="Skip GridSearchCV (use default hyperparameters). Faster for CI.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    args = parser.parse_args()
    run_training(tune=args.tune, cv_folds=args.cv_folds)