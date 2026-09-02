"""
Evaluation metrics for probabilistic classifiers, implemented from scratch
with plain numpy (per the course rule that external libraries may only ease
computation).

Required by the brief: accuracy, log-loss, Brier score. Also included as a
bonus: Expected Calibration Error, and a signed `miscalibration_tilt` used to
identify the *direction* of miscalibration rather than only its size.

Why the three required metrics can disagree, and what each is sensitive to,
is discussed in report/report.tex.
"""

from __future__ import annotations
import numpy as np


def accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """
    Fraction of correct predictions when probabilities are thresholded at
    `threshold`.

    Calibration is monotonic, so it never changes the ranking -- but it can
    change accuracy, by shifting where scores fall relative to the threshold.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return float(np.mean(y_pred == y_true))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Binary cross-entropy; lower is better.

        LL = -(1/n) * sum_i [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]

    Probabilities are clipped to [eps, 1-eps] because a confidently-wrong
    prediction would otherwise give log(0) = -inf. The clip caps the penalty
    at a large finite value, so reported log-losses depend on `eps` whenever
    predictions reach the extremes.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Mean squared error between predicted probability and true label; bounded
    in [0, 1], lower is better.

        BS = (1/n) * sum_i (p_i - y_i)^2

    A model always predicting the base rate scores base_rate * (1 - base_rate).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((p - y_true) ** 2))


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """
    Bin predictions into `n_bins` equal-width bins over [0, 1]; return per
    bin the mean predicted probability, the empirical positive rate, and the
    count. This is the data behind a reliability diagram and behind ECE.

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

    Sample-weighted average absolute gap between the reliability curve and
    the diagonal. 0 is perfect. Bonus diagnostic, not a required metric.
    """
    bin_mean_pred, bin_frac_pos, bin_counts, _ = reliability_bins(y_true, y_prob, n_bins)
    n = np.sum(bin_counts)
    nonempty = bin_counts > 0
    weights = bin_counts[nonempty] / n
    gaps = np.abs(bin_mean_pred[nonempty] - bin_frac_pos[nonempty])
    return float(np.sum(weights * gaps))


def miscalibration_tilt(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Signed measure of the *direction* of miscalibration:

        tilt = weighted mean (pred - obs) over the lower half of [0, 1]
             - weighted mean (pred - obs) over the upper half

    Positive => under-confident (predictions squeezed toward 0.5, reliability
    curve flatter than the diagonal). Negative => over-confident.

    A plain average gap over all bins cannot distinguish the two, since the
    positive gaps at low probabilities cancel the negative ones at high
    probabilities; on synthetic data with a known direction it reads ~0.001
    either way, while this statistic separates them cleanly.

    Noisy on small test sets -- report it across several splits (see
    `src.experiment.run_multi_seed`), not from one.
    """
    bin_mean_pred, bin_frac_pos, bin_counts, _ = reliability_bins(y_true, y_prob, n_bins)
    idx = np.arange(n_bins)
    lower = (idx < n_bins // 2) & (bin_counts > 0)
    upper = (idx >= n_bins // 2) & (bin_counts > 0)
    if bin_counts[lower].sum() == 0 or bin_counts[upper].sum() == 0:
        return float("nan")
    gap = bin_mean_pred - bin_frac_pos
    lower_gap = np.average(gap[lower], weights=bin_counts[lower])
    upper_gap = np.average(gap[upper], weights=bin_counts[upper])
    return float(lower_gap - upper_gap)


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, threshold: float = 0.5) -> dict:
    """Convenience wrapper: compute all metrics at once and return a dict."""
    return {
        "accuracy": accuracy(y_true, y_prob, threshold=threshold),
        "log_loss": log_loss(y_true, y_prob),
        "brier_score": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "tilt": miscalibration_tilt(y_true, y_prob, n_bins=n_bins),
    }
