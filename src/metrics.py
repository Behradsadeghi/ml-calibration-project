"""
Evaluation metrics for probabilistic classifiers, implemented from scratch
with plain numpy (per the assignment's "external libraries only to ease
computation" rule).

Three metrics are required by the assignment brief:

    1. Accuracy   -- fraction of correctly classified examples at a 0.5
                      probability threshold. This only looks at the final
                      hard decision, so it is blind to calibration quality:
                      a model can have perfect accuracy while being wildly
                      over- or under-confident about *how sure* it is.

    2. Log-loss   -- (negative log-likelihood / cross-entropy) the proper
                      scoring rule that logistic regression itself
                      optimizes. It heavily penalizes confident-and-wrong
                      predictions (as p -> 0 or 1 on the wrong side, the
                      loss -> infinity), so it is very sensitive to bad
                      calibration at the extremes.

    3. Brier score -- mean squared error between predicted probability and
                       the true 0/1 label. Like log-loss it is a *proper
                       scoring rule* (you cannot game it by reporting
                       anything other than your true belief), but it is
                       bounded in [0, 1] and penalizes extreme mistakes
                       less harshly than log-loss (quadratically instead of
                       logarithmically).

We also implement Expected Calibration Error (ECE), which is not required
by the brief but is the standard scalar summary of a reliability diagram
(it is literally the weighted-average vertical gap between the calibration
curve and the diagonal). It is included as a bonus diagnostic to make the
before/after discussion concrete without having to eyeball plots.
"""

from __future__ import annotations
import numpy as np


def accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """
    Fraction of correct predictions when probabilities are thresholded at
    `threshold` to obtain hard class labels.

    Note: because Platt scaling and isotonic regression are monotonic
    (rank-preserving) transforms of the raw score, they do NOT change which
    example ranks above which -- but they CAN change accuracy, because they
    shift *where* the 0.5 decision boundary falls relative to each example's
    score. This is why accuracy before/after calibration is not identical
    even though both models predict the same relative ordering.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return float(np.mean(y_pred == y_true))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Binary cross-entropy:

        LL = -(1/n) * sum_i [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]

    Lower is better. Probabilities are clipped to [eps, 1-eps] first,
    because a single confidently-wrong prediction (p=0 when y=1, or vice
    versa) would otherwise produce log(0) = -infinity and blow up the whole
    metric -- clipping caps the penalty at a large-but-finite value instead.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Mean squared error between predicted probability and true label:

        BS = (1/n) * sum_i (p_i - y_i)^2

    Bounded in [0, 1]; lower is better. A model that always predicts the
    base rate gets BS = base_rate * (1 - base_rate), a useful reference
    point ("climatology" baseline in the forecasting literature).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((p - y_true) ** 2))


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """
    Bin predictions into `n_bins` equal-width bins over [0, 1] and compute,
    per bin: the mean predicted probability, the empirical fraction of
    positives, and the number of points. This is the raw data behind a
    reliability diagram (see src/plotting.py) and behind ECE below.

    Returns
    -------
    bin_mean_pred : (n_bins,) array, NaN for empty bins
    bin_frac_pos  : (n_bins,) array, NaN for empty bins
    bin_counts    : (n_bins,) array of ints
    bin_edges     : (n_bins+1,) array of bin boundaries
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize returns 1..n_bins for values in (edges[i-1], edges[i]];
    # subtract 1 to get 0-indexed bins, and clip so p=0 falls in bin 0.
    bin_idx = np.clip(np.digitize(p, bin_edges[1:-1], right=True), 0, n_bins - 1)

    bin_mean_pred = np.full(n_bins, np.nan)
    bin_frac_pos = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_idx == b
        count = int(np.sum(mask))
        bin_counts[b] = count
        if count > 0:
            bin_mean_pred[b] = float(np.mean(p[mask]))
            bin_frac_pos[b] = float(np.mean(y_true[mask]))

    return bin_mean_pred, bin_frac_pos, bin_counts, bin_edges


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum_b (n_b / n) * | mean_pred(b) - frac_pos(b) |

    The sample-weighted average absolute gap between the reliability curve
    and the diagonal (perfect calibration). A single number summarizing
    "how miscalibrated is this model on average" -- 0 is perfect, larger is
    worse. Bonus diagnostic, not one of the three required metrics.
    """
    bin_mean_pred, bin_frac_pos, bin_counts, _ = reliability_bins(y_true, y_prob, n_bins)
    n = np.sum(bin_counts)
    nonempty = bin_counts > 0
    weights = bin_counts[nonempty] / n
    gaps = np.abs(bin_mean_pred[nonempty] - bin_frac_pos[nonempty])
    return float(np.sum(weights * gaps))


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, threshold: float = 0.5) -> dict:
    """Convenience wrapper: compute all metrics at once and return a dict."""
    return {
        "accuracy": accuracy(y_true, y_prob, threshold=threshold),
        "log_loss": log_loss(y_true, y_prob),
        "brier_score": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
    }
