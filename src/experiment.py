"""
End-to-end experiment pipeline for the calibration study.

For each dataset, this module:
  1. Loads the data and splits it into train / calibration / test (see
     `src.datasets.split_train_calib_test` for why a *separate* calibration
     set is required).
  2. Trains Logistic Regression and Random Forest on the training set
     (off-the-shelf sklearn models, as explicitly permitted by the
     assignment for these two).
  3. Reads off each model's raw (uncalibrated) predicted probabilities.
  4. Fits our from-scratch Platt scaling and isotonic regression (PAVA) on
     the calibration set only, then applies them to the test set.
  5. Evaluates accuracy / log-loss / Brier score / ECE on the test set for
     the uncalibrated model and both calibrated versions.
  6. Saves a metrics table (CSV) and a reliability-diagram figure (PNG,
     uncalibrated vs. Platt vs. isotonic overlaid) per dataset x model.

After both datasets have been run, `main()` also builds the mandatory
across-dataset comparison table (results/metrics/summary_comparison.csv).

Run with:  python -m src.experiment   (from the repo root, inside the venv)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .datasets import load_breast_cancer_dataset, load_diabetes_dataset, split_train_calib_test
from .calibration import PlattScaling, IsotonicRegressionPAVA
from .metrics import evaluate_probs
from .plotting import plot_reliability_comparison

SEED = 42
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"


def train_base_models(X_train: np.ndarray, y_train: np.ndarray, seed: int = SEED) -> dict:
    """
    Train the two off-the-shelf base classifiers. Both are used purely as
    "sources of imperfectly-calibrated probabilities" for the calibration
    study -- their own hyperparameters are not the object of study, so we
    use reasonable defaults rather than tuning them.

    Logistic Regression is expected to be reasonably well-calibrated out of
    the box, because it is fit by maximizing log-likelihood, i.e. it is
    already directly optimizing (a version of) the log-loss metric we use
    to evaluate calibration.

    Random Forest averages the (0/1-ish) votes of many de-correlated trees.
    Niculescu-Mizil & Caruana (2005) show this averaging systematically
    pushes predicted probabilities toward 0.5 -- individual trees are
    overconfident, but averaging many of them washes out the extremes, so
    the ensemble ends up *under-confident* near 0 and 1. n_estimators=300
    gives a stable ensemble while keeping runtime small.
    """
    logreg = LogisticRegression(max_iter=2000, random_state=seed)
    logreg.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)

    return {"LogisticRegression": logreg, "RandomForest": rf}


def run_dataset_experiment(load_fn, n_bins: int = 10, seed: int = SEED) -> pd.DataFrame:
    X, y, name, description = load_fn()
    splits = split_train_calib_test(X, y, seed=seed)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_calib, y_calib = splits["X_calib"], splits["y_calib"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    # Standardize features (fit on train only). This matters for Logistic
    # Regression's optimizer (features on very different scales, e.g. the
    # breast-cancer dataset, slow/prevent convergence) and is a no-op for
    # Random Forest's predictions (tree splits are invariant to monotonic
    # per-feature transforms like standardization), so it is safe to apply
    # uniformly to both models.
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_calib = scaler.transform(X_calib)
    X_test = scaler.transform(X_test)

    print(f"[{name}] {description}")
    print(
        f"[{name}] train={len(y_train)} calib={len(y_calib)} test={len(y_test)} "
        f"pos_rate(train/calib/test)="
        f"{y_train.mean():.3f}/{y_calib.mean():.3f}/{y_test.mean():.3f}"
    )

    models = train_base_models(X_train, y_train, seed=seed)

    rows = []
    for model_name, model in models.items():
        # Raw, uncalibrated probability estimates from the base model.
        # These are the "scores" fed to both calibrators: Platt scaling
        # fits a sigmoid on top of them, isotonic regression fits a
        # monotonic step function on top of them. Fitting happens ONLY on
        # the calibration split.
        p_calib_raw = model.predict_proba(X_calib)[:, 1]
        p_test_raw = model.predict_proba(X_test)[:, 1]

        platt = PlattScaling().fit(p_calib_raw, y_calib)
        p_test_platt = platt.predict(p_test_raw)

        isotonic = IsotonicRegressionPAVA().fit(p_calib_raw, y_calib)
        p_test_iso = isotonic.predict(p_test_raw)

        prob_variants = {
            "Uncalibrated": p_test_raw,
            "Platt": p_test_platt,
            "Isotonic": p_test_iso,
        }
        for method_name, p in prob_variants.items():
            m = evaluate_probs(y_test, p, n_bins=n_bins)
            rows.append({"dataset": name, "model": model_name, "method": method_name, **m})

        plot_path = PLOTS_DIR / f"{name}_{model_name}_reliability.png"
        plot_reliability_comparison(
            y_test, prob_variants, n_bins=n_bins,
            title=f"{name} — {model_name} (test set)", save_path=plot_path,
        )
        print(f"[{name}] saved plot -> {plot_path.relative_to(REPO_ROOT)}")

    df = pd.DataFrame(rows)
    csv_path = METRICS_DIR / f"{name}_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[{name}] saved metrics -> {csv_path.relative_to(REPO_ROOT)}")
    return df


def build_cross_dataset_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    The assignment requires a comparison ACROSS the two datasets. This
    builds one table with, for every (dataset, model, method) row, the
    raw metrics plus the change in log-loss / Brier / ECE relative to that
    same dataset+model's uncalibrated baseline -- so improvements are
    directly comparable side by side across datasets even though the two
    datasets have very different absolute difficulty.
    """
    base = df_all[df_all.method == "Uncalibrated"].set_index(["dataset", "model"])
    rows = []
    for (dataset, model), g in df_all.groupby(["dataset", "model"]):
        b = base.loc[(dataset, model)]
        for _, r in g.iterrows():
            rows.append({
                "dataset": dataset, "model": model, "method": r.method,
                "accuracy": r.accuracy, "log_loss": r.log_loss,
                "brier_score": r.brier_score, "ece": r.ece,
                "delta_log_loss_vs_uncal": r.log_loss - b.log_loss,
                "delta_brier_vs_uncal": r.brier_score - b.brier_score,
                "delta_ece_vs_uncal": r.ece - b.ece,
            })
    return pd.DataFrame(rows)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df_bc = run_dataset_experiment(load_breast_cancer_dataset)
    df_db = run_dataset_experiment(load_diabetes_dataset)

    df_all = pd.concat([df_bc, df_db], ignore_index=True)
    df_all.to_csv(METRICS_DIR / "all_results.csv", index=False)

    summary = build_cross_dataset_summary(df_all)
    summary.to_csv(METRICS_DIR / "summary_comparison.csv", index=False)
    print(f"[summary] saved -> results/metrics/summary_comparison.csv")

    pd.set_option("display.width", 120)
    print("\n=== Test-set metrics, all dataset x model x method combinations ===")
    print(
        df_all.pivot_table(
            index=["dataset", "model"], columns="method",
            values=["log_loss", "brier_score", "ece", "accuracy"],
        ).round(4)
    )
    return df_all, summary


if __name__ == "__main__":
    main()
