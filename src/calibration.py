"""
From-scratch implementations of the two calibration methods studied in
Niculescu-Mizil & Caruana (2005): Platt scaling (a 1-D logistic fit on the
raw scores) and isotonic regression via PAVA (a free-form monotonic fit).

Both classes share the interface `fit(scores, y)` / `predict(scores)`, where
`scores` are a base classifier's uncalibrated outputs and `y` are binary
labels in {0, 1}.

Course rule: plain numpy only -- no sklearn.calibration, no scipy.optimize.
The Platt optimisation is a hand-written Newton's method.

The derivations, the comparison between the two methods, and the analysis of
where each one fails are in report/report.tex; the docstrings here cover only
what is needed to read the code.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# 1. Platt Scaling
# ---------------------------------------------------------------------------
class PlattScaling:
    """
    Fits a sigmoid  P(y=1 | f) = 1 / (1 + exp(-(A*f + B)))  on a classifier's
    raw score `f`, by maximum likelihood via Newton's method.

    Two parameters only, which makes the method data-efficient on small
    calibration sets but limits it to monotonic, sigmoid-shaped corrections.

    Sign convention: Platt (1999) writes this as 1 / (1 + exp(A*f + B)), in
    which a good fit has A negative. This uses the standard logistic form
    with the negation inside, so a good fit has A positive (~ +5 here). The
    two are equivalent under A -> -A, B -> -B.

    Regression targets are Bayesian-smoothed rather than the raw {0, 1}
    labels:

        t_i = (N+ + 1) / (N+ + 2)   if y_i = 1
        t_i = 1 / (N- + 2)          if y_i = 0

    On perfectly separable calibration data, raw labels would drive
    |A| -> infinity and collapse the sigmoid into a step function. Smoothing
    bounds the targets away from 0 and 1, so the fitted probability can never
    reach either. See report/report.tex, section on Platt scaling.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-8):
        self.max_iter = max_iter
        self.tol = tol
        self.A_: float | None = None
        self.B_: float | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable logistic sigmoid: avoids overflow in exp(z)
        # for very negative or very positive z.
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
        return out

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattScaling":
        """
        Fit A, B via Newton's method on the calibration set.

        Parameters
        ----------
        scores : (n,) array of raw, uncalibrated model outputs.
        y      : (n,) array of true binary labels in {0, 1}.
        """
        f = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        n_pos = np.sum(y == 1)
        n_neg = n - n_pos

        # Bayesian-smoothed regression targets (see docstring above).
        t = np.where(y == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))

        # Initialize at the "no-op" sigmoid: A=0, B = log-odds of the
        # smoothed base rate, so predictions start as a constant equal to
        # the calibration set's positive rate.
        A, B = 0.0, np.log((n_neg + 1.0) / (n_pos + 1.0))

        prev_nll = np.inf
        for _ in range(self.max_iter):
            z = A * f + B
            p = self._sigmoid(z)
            p = np.clip(p, 1e-12, 1 - 1e-12)  # guard log(0)

            # Negative log-likelihood (for convergence monitoring).
            nll = -np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))

            # Gradient of NLL w.r.t. (A, B).
            d = p - t
            grad_A = np.sum(d * f)
            grad_B = np.sum(d)

            # Hessian of NLL w.r.t. (A, B): H = J^T W J with W = diag(p(1-p)).
            w = p * (1 - p)
            h_AA = np.sum(w * f * f) + 1e-12  # tiny ridge for stability
            h_AB = np.sum(w * f)
            h_BB = np.sum(w) + 1e-12

            det = h_AA * h_BB - h_AB * h_AB
            if abs(det) < 1e-12:
                break  # Hessian (near-)singular: stop, keep current A,B

            # Newton step: solve H * [dA, dB]^T = -grad  via the closed-form
            # inverse of a 2x2 matrix.
            dA = -(h_BB * grad_A - h_AB * grad_B) / det
            dB = -(-h_AB * grad_A + h_AA * grad_B) / det

            A += dA
            B += dB

            if abs(prev_nll - nll) < self.tol:
                break
            prev_nll = nll

        self.A_, self.B_ = A, B
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Return calibrated P(y=1 | score) for new raw scores."""
        if self.A_ is None:
            raise RuntimeError("PlattScaling must be fit() before predict().")
        f = np.asarray(scores, dtype=np.float64)
        return self._sigmoid(self.A_ * f + self.B_)


# ---------------------------------------------------------------------------
# 2. Isotonic Regression via the Pool-Adjacent-Violators Algorithm (PAVA)
# ---------------------------------------------------------------------------
class IsotonicRegressionPAVA:
    """
    Non-parametric calibration: the non-decreasing g(f) minimising
    sum_i (y_i - g(f_i))^2, found with the Pool-Adjacent-Violators Algorithm.

    PAVA sorts by score, starts one block per point, and repeatedly merges
    adjacent blocks whose values violate monotonicity (re-checking backwards
    after each merge, since a merge can create a new violation to its left).
    The resulting blocks are the exact least-squares solution.

    Unlike Platt's fixed sigmoid this can represent any monotonic map, at the
    cost of far more effective degrees of freedom -- so it needs a larger
    calibration set to avoid overfitting.

    Prediction interpolates linearly between block means, flat beyond the
    observed range. Two consequences worth knowing:

    - This is a smoothing choice, not the argmin: interpolating between block
      *means* means predictions at training points inside a block differ from
      that block's fitted constant. Scikit-learn interpolates between block
      *boundaries* and does reproduce the exact values.
    - On a small, perfectly rank-separated calibration set PAVA finds no
      violations, so every point stays its own block and every block value is
      exactly 0 or 1 -- the labels are memorised. Nothing here plays the role
      of Platt's smoothed targets. Predictions then collapse onto {0, 1} and
      log-loss (unbounded) is punished severely while Brier and accuracy
      barely move.

    Both points are analysed with numbers in report/report.tex.
    """

    def __init__(self):
        self.x_thresholds_: np.ndarray | None = None
        self.y_values_: np.ndarray | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "IsotonicRegressionPAVA":
        """
        Run PAVA on the calibration set.

        Parameters
        ----------
        scores : (n,) array of raw, uncalibrated model outputs.
        y      : (n,) array of true binary labels in {0, 1}.
        """
        f = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        order = np.argsort(f, kind="mergesort")  # stable sort
        f_sorted = f[order]
        y_sorted = y[order]

        # Each block is (sum_x, sum_y, weight) so we can recompute means
        # cheaply after merges. We use a simple stack-based PAVA pass:
        # O(n) amortized because each point is pushed and popped at most
        # once.
        block_sum_x = []   # sum of scores in block
        block_sum_y = []   # sum of labels in block
        block_w = []       # number of points in block

        for xi, yi in zip(f_sorted, y_sorted):
            block_sum_x.append(xi)
            block_sum_y.append(yi)
            block_w.append(1.0)

            # Pool backwards while the last two blocks violate monotonicity.
            while len(block_w) > 1 and (block_sum_y[-2] / block_w[-2]) > (
                block_sum_y[-1] / block_w[-1]
            ):
                sx = block_sum_x.pop()
                sy = block_sum_y.pop()
                w = block_w.pop()
                block_sum_x[-1] += sx
                block_sum_y[-1] += sy
                block_w[-1] += w

        block_sum_x = np.array(block_sum_x)
        block_sum_y = np.array(block_sum_y)
        block_w = np.array(block_w)

        # Representative point of each block = its mean score; fitted
        # value = its mean label (the pooled positive rate).
        self.x_thresholds_ = block_sum_x / block_w
        self.y_values_ = block_sum_y / block_w
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Return calibrated P(y=1 | score) for new raw scores via linear
        interpolation between the fitted blocks (flat beyond the observed
        range)."""
        if self.x_thresholds_ is None:
            raise RuntimeError("IsotonicRegressionPAVA must be fit() before predict().")
        f = np.asarray(scores, dtype=np.float64)
        # np.interp already clips to the boundary y-values outside the
        # range of x_thresholds_, which is exactly the "flat extrapolation"
        # behavior we want.
        return np.interp(f, self.x_thresholds_, self.y_values_)
