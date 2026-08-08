# Assignment 2 — Calibration of Probabilities

Statistical Methods for Machine Learning (University of Milan).
Reference paper: Niculescu-Mizil & Caruana, *"Predicting Good Probabilities with Supervised Learning"*, ICML 2005.

Evaluates and improves the calibration of Logistic Regression and Random Forest classifiers on two
real-world datasets, using from-scratch implementations of Platt scaling and isotonic regression (PAVA).

## Repository structure

```
data/                 # dataset cache (populated on first run by sklearn.datasets)
src/
  calibration.py       # Platt scaling + isotonic regression (PAVA), from scratch, numpy only
  metrics.py            # accuracy, log-loss, Brier score, ECE, reliability-diagram binning — from scratch
  datasets.py            # dataset loading + stratified train/calibration/test split
  plotting.py              # reliability diagram plotting (matplotlib)
  experiment.py              # orchestrates training, calibration, evaluation, saves results/
notebooks/
  01_breast_cancer.ipynb      # experiment + discussion for dataset 1
  02_diabetes.ipynb            # experiment + discussion for dataset 2
  03_cross_dataset_summary.ipynb  # mandatory across-dataset comparison
report/                # LaTeX report (to be added later)
results/
  metrics/              # CSV metrics tables (per-dataset + combined + cross-dataset summary)
  plots/                 # reliability diagrams (PNG) + cross-dataset summary bar chart
```

## Reproducing the results

Everything is pinned and seeded for reproducibility (`SEED = 42` in `src/experiment.py`, used for the
train/calibration/test split and both models).

```bash
cd calibration-assignment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the full pipeline for both datasets: trains models, calibrates, evaluates,
# saves results/metrics/*.csv and results/plots/*.png
python -m src.experiment
```

The diabetes dataset is fetched once via `sklearn.datasets.fetch_openml` and cached locally
(`~/scikit_learn_data` by default) — after the first run, everything is fully offline.

To explore interactively / regenerate the notebook outputs:

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
# or: jupyter notebook  (and run cells manually)
```

## Datasets and why they were chosen

The assignment requires two real-world classification datasets that let us compare calibration
behavior across different data regimes. We picked datasets that differ in the two properties that most
directly drive miscalibration: **class separability** and **class balance**.

| | Breast Cancer Wisconsin (Diagnostic) | Pima Indians Diabetes |
|---|---|---|
| Source | `sklearn.datasets.load_breast_cancer` | `sklearn.datasets.fetch_openml(data_id=37)` (UCI mirror) |
| Samples / features | 569 / 30 | 768 / 8 |
| Positive rate | ~37% (malignant) | ~35% (tested positive) |
| Separability | High — linear model reaches ~97% test accuracy | Moderate — noisier, ~70-75% test accuracy ceiling |

- **Breast Cancer** is an "easy", well-separated problem: we expect both models to already be close to
  well-calibrated, so it is a useful low-miscalibration reference point.
- **Diabetes** is noisier and less separable, so we expect more miscalibration to start with, and more
  room for the calibration methods to actually help.

Both are small-to-medium scale (as required) and binary classification.

## Methodology notes

- **Train / calibration / test split**: stratified 60/20/20 split (`src/datasets.split_train_calib_test`).
  Platt scaling and isotonic regression are fit **only** on the calibration split — never on the test
  set (that would leak test labels into the calibrated model) and never on the training split (that
  would just correct the base model's training-set overconfidence, not its true generalization
  behavior).
- **Feature scaling**: `StandardScaler`, fit on the training split only. This is needed for Logistic
  Regression's optimizer to converge reliably (breast-cancer features span several orders of magnitude)
  and is a no-op for Random Forest's predictions (tree splits are invariant to monotonic per-feature
  transforms).
- **Diabetes missing-value quirk**: several features use `0` as a placeholder for a missing measurement
  (not physiologically valid for e.g. blood pressure or BMI). These are treated as `NaN` and
  median-imputed using statistics from the training split only.
- **What "raw score" means for calibration**: both calibrators are fit on top of each model's raw
  `predict_proba` output (not `decision_function`), so the two calibration methods are applied
  uniformly to both models even though Random Forest has no natural decision-function margin.
- **From-scratch vs. off-the-shelf**: per the course rule, Logistic Regression and Random Forest are
  used off-the-shelf (scikit-learn); Platt scaling and isotonic regression (PAVA) are implemented from
  scratch in `src/calibration.py` using only numpy, with the math explained in the docstrings. Accuracy,
  log-loss, Brier score, and the reliability-diagram binning are also implemented from scratch in
  `src/metrics.py`, since they are not explicitly whitelisted as off-the-shelf either.
- **Accuracy before/after calibration**: because Platt scaling and isotonic regression are monotonic
  (rank-preserving) transforms, they never change AUC/ranking — but they *can* change 0.5-threshold
  accuracy, because they shift where each example's score lands relative to 0.5. Don't be surprised if
  accuracy moves slightly after calibration.

## Known finding worth understanding for the oral exam

On the Breast Cancer dataset, the Logistic Regression calibration set (114 points) turns out to be
**perfectly rank-separated** by the model's raw scores — PAVA finds zero monotonicity violations to
pool, so isotonic regression degenerates into memorizing the calibration labels as a hard 0/1 step
function. A misclassified test point then gets predicted probability exactly 0 or 1, which is
catastrophic for log-loss (eps-clipped to a large finite penalty) even though it barely moves Brier
score or accuracy. Platt scaling's Bayesian-smoothed regression targets make it structurally immune to
this failure mode (see the extended docstring in `src/calibration.py`). This is a genuine, reproducible
result — not a bug — and a good concrete example of the "isotonic regression needs more calibration
data than Platt scaling" tradeoff discussed in Niculescu-Mizil & Caruana (2005).

## Extra (bonus) diagnostic: Expected Calibration Error (ECE)

Beyond the three required metrics (accuracy, log-loss, Brier score), `src/metrics.py` also computes ECE
— the sample-weighted average gap between the reliability curve and the diagonal — as a single-number
summary that makes the before/after discussion concrete without having to eyeball plots. It is not a
required metric, just a convenience.

## Status

Code, experiments, metrics, and plots are complete and reproducible. The LaTeX report in `report/` has
not been written yet (by design — plots and result tables above are meant to be the input to it).
