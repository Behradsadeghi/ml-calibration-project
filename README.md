# Assignment 2 — Calibration of Probabilities

Statistical Methods for Machine Learning (University of Milan).
Reference paper: Niculescu-Mizil & Caruana, *"Predicting Good Probabilities with Supervised Learning"*, ICML 2005.

Evaluates and improves the calibration of Logistic Regression and Random Forest classifiers on two
real-world datasets, using from-scratch implementations of Platt scaling and isotonic regression (PAVA).

## Repository structure

```
data/
  diabetes.csv           # committed offline copy of the raw OpenML diabetes fetch (see below)
src/
  calibration.py       # Platt scaling + isotonic regression (PAVA), from scratch, numpy only
  metrics.py            # accuracy, log-loss, Brier score, ECE, reliability-diagram binning — from scratch
  datasets.py            # dataset loading + stratified train/calibration/test split
  plotting.py              # reliability diagram plotting (matplotlib)
  experiment.py              # orchestrates training, calibration, evaluation, saves results/
                              # (also: run_multi_seed, the multi-seed robustness check)
notebooks/
  01_breast_cancer.ipynb      # experiment + discussion for dataset 1
  02_diabetes.ipynb            # experiment + discussion for dataset 2
  03_cross_dataset_summary.ipynb  # mandatory across-dataset comparison + multi-seed robustness check
report/                # LaTeX report
  report.tex             # source (figures referenced from ../results/plots/)
  report.pdf              # compiled PDF: pdflatex report.tex (twice, to resolve references)
results/
  metrics/              # CSV metrics tables: per-dataset, combined, cross-dataset summary,
                          # multiseed_raw.csv / multiseed_summary.csv
  plots/                 # reliability diagrams (PNG), cross-dataset summary bar chart,
                          # multiseed_stability.png
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

# Also run the multi-seed robustness check (10 seeds x 2 datasets, ~10s):
# saves results/metrics/multiseed_{raw,summary}.csv and results/plots/multiseed_stability.png
python -m src.experiment --multiseed        # or: --multiseed --n-seeds 20
```

**The diabetes dataset reproduces fully offline.** `data/diabetes.csv` is a committed copy of the raw
data from `sklearn.datasets.fetch_openml(data_id=37)` (the OpenML mirror of the UCI Pima Indians
Diabetes dataset). `src/datasets.load_diabetes_dataset` loads from this file if it exists, and only
falls back to `fetch_openml` (which requires internet on first use, then caches to
`~/scikit_learn_data`) if it does not — e.g. if you delete `data/diabetes.csv` to force a fresh fetch.
Either path produces the identical raw data, since the CSV was written directly from the first
successful fetch. Breast Cancer is bundled with scikit-learn and is always offline.

Note also that `requirements.txt` pins exact library versions. This matters: Random Forest results shift
slightly across scikit-learn versions even with a fixed `random_state`, so reproducing the exact numbers
in `results/` requires the pinned versions.

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
- **Diabetes** is noisier and less separable, so we *expected* more miscalibration to start with, and
  more room for the calibration methods to help. This expectation was only partly borne out: the
  uncalibrated models are indeed less sharp here, but neither calibration method improves log-loss over
  the uncalibrated baseline on this dataset (see the multi-seed table below). Reporting the expectation
  and where it failed is part of the analysis, not a defect in it.

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
pool, so every calibration point stays in its own block and every fitted block value is exactly 0 or 1.
Isotonic regression has effectively memorized the calibration labels.

Predictions are still piecewise-linear (we interpolate between block means), but since all block values
are 0 or 1, and `np.interp` clamps flat outside the observed score range, **113 of the 114 test
predictions come out at exactly 0 or 1**. Two of those are misclassified, and that alone drives test
log-loss from 0.086 (uncalibrated) to 0.620 — while Brier score (0.0239 → 0.0230) and accuracy
(0.965 → 0.974) barely move at all.

Two caveats to state honestly when discussing this:

- **The magnitude depends on the clipping constant.** Log-loss is eps-clipped in `src/metrics.py`
  (otherwise it would be infinite). At `eps=1e-15` that run gives 0.620; at `eps=1e-3` the same
  predictions give 0.136. The *direction* is robust, the specific number is not — always report the eps.
- **The effect is reproducible, but high-variance, and not unique to Breast Cancer.** See the
  "Multi-seed robustness check" section below for the formal version of this claim (10 seeds, both
  datasets) — isotonic's log-loss is consistently higher than Platt's on *both* datasets, with a
  standard deviation on Breast Cancer large enough that no single split's number should be
  over-interpreted.

Platt scaling's Bayesian-smoothed regression targets make it structurally immune to this failure mode
(see the extended docstring in `src/calibration.py`). This is a genuine result — not a bug — and a good
concrete example of the "isotonic regression needs more calibration data than Platt scaling" tradeoff
discussed in Niculescu-Mizil & Caruana (2005).

## Multi-seed robustness check

Every number above comes from a single random train/calibration/test split (`SEED = 42`). With
calibration/test sets of only ~114-154 points, that is one sample from "what would happen with a
different split," not necessarily a stable estimate. `src.experiment.run_multi_seed` reruns the entire
pipeline (fresh split, fresh model fit, fresh calibration fit, fresh evaluation) across `N=10` seeds
(42-51 by default) for both datasets and reports test-metric mean ± std per (dataset, model, method).
It is purely additive — it does not touch any single-seed output — and is invoked with:

```bash
python -m src.experiment --multiseed        # writes results/metrics/multiseed_{raw,summary}.csv
                                              # and results/plots/multiseed_stability.png
```

Mean test log-loss ± std across 10 seeds (the metric most affected by isotonic's 0/1-collapse):

| dataset | model | Uncalibrated | Platt | Isotonic |
|---|---|---|---|---|
| breast_cancer | LogisticRegression | 0.085 ± 0.029 | **0.086 ± 0.027** | 0.350 ± 0.241 |
| breast_cancer | RandomForest | 0.156 ± 0.111 | **0.124 ± 0.045** | 0.409 ± 0.365 |
| diabetes | LogisticRegression | 0.487 ± 0.038 | **0.493 ± 0.035** | 0.803 ± 0.344 |
| diabetes | RandomForest | 0.481 ± 0.034 | **0.490 ± 0.034** | 0.818 ± 0.242 |

Two conclusions follow, and they revise an earlier (single-seed) claim in `notebooks/02_diabetes.ipynb`
that isotonic regression "has more room to pay off" on the noisier diabetes dataset — that claim did
not survive multi-seed evidence and has been corrected there:

1. **The direction is real and dataset-independent.** Isotonic's mean log-loss is higher than Platt's
   in all four (dataset, model) combinations, not only on the well-separated Breast Cancer dataset. The
   effect is smaller in *relative* terms on Diabetes (mean ~1.6x Platt's, vs. ~3-4x on Breast Cancer),
   but it does not disappear.
2. **Isotonic's variance is large enough that single-split numbers should not be over-interpreted** —
   on Breast Cancer / Random Forest its std (0.365) is nearly as large as its own mean (0.409). Brier
   score and ECE, being bounded metrics that cannot blow up near p=0/1, do not show this instability;
   see `results/plots/multiseed_stability.png` (also reproduced in `notebooks/03_cross_dataset_summary.ipynb`).

**Practical takeaway:** at calibration-set sizes this small (~100-150 points), Platt scaling is the
safer default for both datasets studied here. Isotonic regression's extra flexibility is a genuine
theoretical advantage (see its docstring), but realizing it reliably needs more calibration data than
either dataset's split provides.

## Did calibration actually help?

The headline finding above is about isotonic regression failing. But the prior question — *did either
calibration method improve on the uncalibrated baseline at all?* — has a more nuanced answer, and it is
worth stating plainly. Using the 10-seed means, comparing Platt scaling against the uncalibrated model:

| dataset | model | metric | Uncalibrated | Platt | verdict |
|---|---|---|---|---|---|
| breast_cancer | LogisticRegression | log-loss | 0.0846 | 0.0864 | worse |
| breast_cancer | RandomForest | log-loss | 0.1563 | **0.1235** | **better** |
| diabetes | LogisticRegression | log-loss | 0.4868 | 0.4930 | worse |
| diabetes | RandomForest | log-loss | 0.4810 | 0.4896 | worse |
| breast_cancer | LogisticRegression | ECE | 0.0319 | **0.0291** | **better** |
| breast_cancer | RandomForest | ECE | 0.0545 | **0.0476** | **better** |
| diabetes | LogisticRegression | ECE | 0.0850 | **0.0805** | **better** |
| diabetes | RandomForest | ECE | 0.0724 | 0.0850 | worse |

So Platt scaling improves ECE in 3 of 4 cases, Brier score in 2 of 4, and log-loss in only 1 of 4. The
single clear win — Random Forest on Breast Cancer — is *not* the case with the largest raw miscalibration
(Diabetes has higher uncalibrated ECE for both models). It is the case whose miscalibration is most
**systematic**: the largest tilt with a stable sign across splits (see the tilt table below). Platt scaling
has two parameters and can only undo a distortion with a consistent sigmoid shape — it cannot repair
miscalibration that has no stable direction. On Diabetes the tilt's sign flips from split to split, so
there is more error to fix but no systematic shape to fix it with.

**Why so little to fix.** This is the expected result for these two model classes, not a failure of the
implementation:

- Logistic Regression is fit by maximizing log-likelihood, i.e. it directly optimizes (a regularized
  version of) the very loss used to evaluate calibration. On well-specified problems it is already
  close to calibrated, so post-hoc correction has almost nothing to correct and mostly adds estimation
  noise from the ~110-150 point calibration set.
- Random Forest's vote-averaging does push probabilities toward the centre, which is a real, correctable
  distortion — and Platt scaling does correct it, visibly, on Breast Cancer.

**A limitation worth acknowledging.** Niculescu-Mizil & Caruana's most dramatic calibration gains come
from boosted trees, SVMs, and Naive Bayes — model families whose scores are *systematically* distorted
(boosting pushes scores away from 0/1 in a characteristic sigmoidal way; SVM margins are not
probabilities at all). The assignment specifies Logistic Regression and Random Forest, which are two of
the better-calibrated families in that paper's own comparison, so this study is structurally unlikely to
reproduce the paper's largest effects. The results here are consistent with the paper — they sit at the
"already nearly calibrated" end of its spectrum — rather than contradicting it.

## Which model is more miscalibrated, and in which direction?

The assignment asks for a discussion of miscalibration, which means answering not just *how much* but
*which way*. `src/metrics.miscalibration_tilt` measures the direction: it is the weighted mean gap
(predicted − observed) over the lower half of the probability range minus the same over the upper half.

- **Positive tilt = under-confident**: predictions squeezed toward 0.5, reliability curve flatter than
  the diagonal. This is the vote-averaging effect Niculescu-Mizil & Caruana describe for tree ensembles.
- **Negative tilt = over-confident**: predictions pushed away from 0.5, curve steeper than the diagonal.

A plain average of (predicted − observed) across all bins *cannot* answer this question, because the
positive gaps at low probabilities and the negative gaps at high probabilities cancel: on synthetic data
with a known direction, that statistic reads ~0.001 for a strongly under-confident model and ~0.001 for
a strongly over-confident one. The tilt statistic separates them cleanly. `miscalibration_tilt` was
validated against exactly such synthetic cases, and `reliability_bins` was cross-checked against
`sklearn.calibration.calibration_curve` (agreement to ~1e-16).

Uncalibrated models, tilt across 10 random splits (from `results/metrics/multiseed_summary.csv`):

| dataset | model | tilt (mean ± std) | range over seeds | reading |
|---|---|---|---|---|
| breast_cancer | LogisticRegression | +0.032 ± 0.019 | −0.007 to +0.056 | under-confident |
| breast_cancer | RandomForest | **+0.059 ± 0.050** | −0.027 to +0.148 | **under-confident, ~2x more than LR** |
| diabetes | LogisticRegression | −0.060 ± 0.115 | −0.229 to +0.125 | no stable direction |
| diabetes | RandomForest | +0.031 ± 0.079 | −0.085 to +0.163 | no stable direction |

**What this supports.** On Breast Cancer the paper's prediction holds: both models are under-confident,
and Random Forest is roughly twice as under-confident as Logistic Regression — consistent with the
vote-averaging mechanism, which applies to the forest and not to the linear model. This is also the one
case where Platt scaling clearly helps (log-loss 0.156 → 0.124), which is what one would expect if
there is a genuine, systematic, sigmoid-shaped distortion for it to undo.

**What this does not support.** On Diabetes the sign flips from split to split for both models, so no
claim about direction can honestly be made there. Reporting a direction from the single `SEED=42` split
would have produced a confident-sounding statement that ten splits do not back up.

**Where the miscalibration actually sits.** Per-bin, the largest gaps on Breast Cancer are in the middle
of the probability range (bins around 0.2–0.7, gaps up to ~0.2–0.3), while the two outermost bins are
close to calibrated (gaps ~0.003–0.016). Those outermost bins hold 77–89% of the test mass on that
dataset, which is why the overall ECE stays small (~0.03–0.05) despite visible curvature in the middle:
the model is miscalibrated mostly where it rarely predicts.

## Extra (bonus) diagnostic: Expected Calibration Error (ECE)

Beyond the three required metrics (accuracy, log-loss, Brier score), `src/metrics.py` also computes ECE
— the sample-weighted average gap between the reliability curve and the diagonal — as a single-number
summary that makes the before/after discussion concrete without having to eyeball plots. It is not a
required metric, just a convenience.

## Status

Code, experiments, metrics, plots (including the multi-seed robustness check), and the LaTeX report are
complete and reproducible, and the diabetes dataset reproduces fully offline via `data/diabetes.csv`.
The report is in `report/` (`report.tex` plus the compiled `report.pdf`); it draws its figures directly
from `results/plots/`, so re-running the pipeline and recompiling keeps the two in sync.
