"""
Dataset loading and train/calibration/test splitting.

The two datasets are chosen to differ in the properties that drive
calibration behaviour -- class separability and balance -- so that the
required across-dataset comparison is informative:

1. Breast Cancer Wisconsin (sklearn) -- 569 x 30, ~37% positive, well
   separated (a linear model reaches ~97% test accuracy). Serves as the
   low-miscalibration reference point.

2. Pima Indians Diabetes (OpenML data_id=37) -- 768 x 8, ~35% positive,
   noisier (~77% accuracy ceiling). Expected to show more miscalibration.
   That expectation was only partly borne out; see report/report.tex.

Both are binary and small-to-medium scale, as required. The diabetes fetch
caches its raw frame to `data/diabetes.csv` (committed) on first success, so
the pipeline reproduces offline after that.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
DIABETES_CSV_PATH = REPO_ROOT / "data" / "diabetes.csv"


def load_breast_cancer_dataset():
    """
    Returns (X, y, name, description). Labels are flipped relative to
    sklearn's encoding so that y=1 means malignant, i.e. the positive class
    is the condition of interest.
    """
    data = load_breast_cancer()
    X = data.data.astype(np.float64)
    y = (1 - data.target).astype(int)  # flip: malignant -> 1, benign -> 0
    name = "breast_cancer"
    description = (
        "Breast Cancer Wisconsin (Diagnostic): 569 samples, 30 features, "
        "binary (1=malignant), ~37% positive rate, well-separated classes."
    )
    return X, y, name, description


def load_diabetes_dataset(cache_dir: str | None = None):
    """
    Returns (X, y, name, description); y=1 means "tested_positive".

    Loads `data/diabetes.csv` if present, so a grader without internet is not
    blocked. Otherwise fetches from OpenML (data_id=37) and writes that same
    raw frame to the CSV, making every later run offline and byte-identical.

    Dataset quirk: glucose, blood pressure, skin thickness, insulin and BMI
    use 0 as a missing-value placeholder, which is not physiologically valid.
    Those zeros are flagged NaN here; imputation happens downstream on the
    training split only, so the leakage boundary stays explicit.
    """
    if DIABETES_CSV_PATH.exists():
        df = pd.read_csv(DIABETES_CSV_PATH)
    else:
        bunch = fetch_openml(data_id=37, as_frame=True, cache=True, data_home=cache_dir)
        df = bunch.frame.copy()
        DIABETES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DIABETES_CSV_PATH, index=False)

    y = (df["class"].astype(str) == "tested_positive").astype(int).to_numpy()
    X_df = df.drop(columns=["class"]).astype(np.float64)

    zero_as_missing_cols = ["plas", "pres", "skin", "insu", "mass"]
    # OpenML's column names for this dataset are abbreviated; fall back to
    # positional handling if names differ across OpenML versions.
    present_cols = [c for c in zero_as_missing_cols if c in X_df.columns]
    if not present_cols and X_df.shape[1] == 8:
        # Fallback: standard Pima column order is
        # [preg, plas, pres, skin, insu, mass, pedi, age]
        present_cols = list(X_df.columns[[1, 2, 3, 4, 5]])

    X = X_df.to_numpy(copy=True)
    col_idx = {c: i for i, c in enumerate(X_df.columns)}
    for c in present_cols:
        i = col_idx[c]
        X[X[:, i] == 0, i] = np.nan

    name = "diabetes"
    description = (
        "Pima Indians Diabetes (OpenML id=37, UCI mirror): 768 samples, 8 "
        "features, binary (1=tested_positive), ~35% positive rate, noisier "
        "and less separable than the breast-cancer dataset."
    )
    return X, y, name, description


def _impute_median(X_train: np.ndarray, *other_arrays: np.ndarray):
    """
    Median-impute NaNs, fitting the median on X_train only and applying it to
    every array passed in. Fitting on train only avoids leakage.
    """
    medians = np.nanmedian(X_train, axis=0)
    outputs = []
    for X in (X_train,) + other_arrays:
        X = X.copy()
        nan_mask = np.isnan(X)
        if nan_mask.any():
            col_idx = np.where(nan_mask.any(axis=0))[0]
            for j in col_idx:
                X[nan_mask[:, j], j] = medians[j]
        outputs.append(X)
    return tuple(outputs)


def split_train_calib_test(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    train_frac: float = 0.6,
    calib_frac: float = 0.2,
    test_frac: float = 0.2,
):
    """
    Stratified three-way split into disjoint train / calibration / test sets.

    The calibration split must be separate from both of the others: fitting a
    calibrator on the training split would correct the base model's
    training-set overconfidence rather than its generalisation behaviour, and
    fitting it on the test split would leak test labels into every reported
    number.

    Stratifying on y keeps the positive rate consistent across the three
    splits, which matters for the imbalanced diabetes data.

    Returns a dict with X_train, y_train, X_calib, y_calib, X_test, y_test.
    NaNs are median-imputed using training-split statistics only.
    """
    assert abs(train_frac + calib_frac + test_frac - 1.0) < 1e-9

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=train_frac, random_state=seed, stratify=y
    )
    # Split the remainder into calibration/test, preserving the requested
    # relative proportions.
    rest_calib_frac = calib_frac / (calib_frac + test_frac)
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_rest, y_rest, train_size=rest_calib_frac, random_state=seed, stratify=y_rest
    )

    if np.isnan(X_train).any() or np.isnan(X_calib).any() or np.isnan(X_test).any():
        X_train, X_calib, X_test = _impute_median(X_train, X_calib, X_test)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_calib": X_calib, "y_calib": y_calib,
        "X_test": X_test, "y_test": y_test,
    }
