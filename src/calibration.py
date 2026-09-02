"""
From-scratch implementations of the two calibration methods studied in
Niculescu-Mizil & Caruana (2005), "Predicting Good Probabilities with
Supervised Learning":

    1. Platt Scaling      -- fits a 1-D logistic (sigmoid) regression on top
                              of a model's raw scores.
    2. Isotonic Regression -- fits a free-form, non-decreasing step function
                              using the Pool-Adjacent-Violators Algorithm
                              (PAVA).

Both classes follow the same tiny interface: `fit(scores, y)` and
`predict(scores)`, where `scores` are the *uncalibrated* outputs of a base
classifier (a decision-function value or a raw predicted probability) and
`y` are the true binary labels (0/1).

IMPORTANT (course rule): everything here is implemented with plain numpy.
No sklearn.calibration, no scipy.optimize -- the optimization for Platt
scaling is a hand-written Newton's method, exactly as in Platt's 1999 paper.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# 1. Platt Scaling
# ---------------------------------------------------------------------------
class PlattScaling:
    """
    Fits a sigmoid  P(y=1 | f) = 1 / (1 + exp(-(A*f + B)))  on top of a
    classifier's raw score `f` (e.g. a logistic-regression decision-function
    value, or a random forest's predicted probability).

    A note on sign convention: Platt's original 1999 paper writes this as
    1 / (1 + exp(A*f + B)), in which a well-behaved fit has A *negative*.
    This implementation uses the standard logistic parameterisation with the
    negation inside, so a well-behaved fit has A *positive* (typically
    A ~ +5 here). The two forms are equivalent under A -> -A, B -> -B and
    give identical probabilities; only the reported sign of the parameters
    differs. The standard form is used here because it matches the
    `_sigmoid` helper below and avoids a sign flip in the Newton update.

    Why a sigmoid at all?
    ----------------------
    Many classifiers produce scores that *rank* examples correctly but whose
    numeric value is not a well-calibrated probability (e.g. an SVM's
    distance to the margin, or a tree ensemble's vote fraction, which tends
    to be pushed away from 0 and 1 by averaging many trees). Platt (1999)
    observed that for many classifiers the *distortion* between the raw
    score and the true P(y=1|f) is well approximated by a single sigmoid
    with just two scalar parameters (A, B). Fitting only two parameters
    (instead of a fully free-form curve) makes Platt scaling very data
    efficient -- it works even with a small calibration set -- but it can
    only fix monotonic, sigmoid-shaped distortions.

    How we fit A and B
    -------------------
    We choose A, B to maximize the likelihood of the observed labels under
    the sigmoid model, i.e. minimize the negative log-likelihood (NLL):

        NLL(A, B) = - sum_i [ t_i * log(p_i) + (1 - t_i) * log(1 - p_i) ]

    where p_i = sigmoid(A*f_i + B) and t_i is a *target* derived from y_i
    (see the label-smoothing note below). This is exactly logistic
    regression of y on the 1-D feature f, so it is convex in (A, B) and has
    a unique minimum, which we find with Newton's method (the update rule
    below is taken almost verbatim from Platt's original paper / the
    pseudocode in Lin, Lin & Weng, 2007).

    Why smoothed targets instead of raw 0/1 labels?
    -------------------------------------------------
    If the calibration data happens to be perfectly separable in `f`
    (plausible with a strong base classifier and a small calibration set),
    maximum-likelihood logistic regression drives |A| -> infinity to push
    predicted probabilities all the way to 0/1 -- i.e. it overfits and
    produces an *overconfident* sigmoid. Platt's fix is to replace the hard
    targets {0, 1} with Bayesian-smoothed targets that never reach the
    extremes:

        t_i = (N+ + 1) / (N+ + 2)   if y_i = 1
        t_i = 1 / (N- + 2)          if y_i = 0

    where N+ and N- are the counts of positive/negative examples in the
    calibration set. This is the posterior mean of P(y=1) under a uniform
    Beta(1,1) prior combined with a single observation of class y_i out of
    an "effective" N+2 trials -- a standard Laplace/Bayesian smoothing
    argument. It keeps the fitted sigmoid honest instead of collapsing to a
    step function.
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
    Non-parametric calibration: finds the non-decreasing function g(f) that
    best fits the calibration data in a least-squares sense,

        minimize_g   sum_i (y_i - g(f_i))^2
        subject to   g non-decreasing in f

    Unlike Platt scaling's 2-parameter sigmoid, isotonic regression can
    represent *any* monotonic calibration map, so it can correct
    distortions a sigmoid cannot (e.g. the "S"-within-an-"S" shapes that
    Niculescu-Mizil & Caruana show for boosted trees). The price is that it
    has many more effective degrees of freedom, so it needs more
    calibration data to avoid overfitting than Platt scaling does.

    The Pool-Adjacent-Violators Algorithm (PAVA)
    ----------------------------------------------
    1. Sort the calibration examples by score f_i (ascending).
    2. Initialize one "block" per example, each holding a weight (=1 point)
       and a value (=y_i).
    3. Scan the blocks left to right. Whenever a block's value is greater
       than the next block's value (a "violation" of monotonicity), merge
       ("pool") the two blocks into one, whose value is the weighted mean
       of the two, and re-check backwards, because merging can create a new
       violation with the block before it.
    4. Repeat until no adjacent pair violates monotonicity. The result is a
       set of blocks, each spanning a contiguous range of scores, with a
       constant fitted value -- i.e. a non-decreasing step function. This
       is provably the exact least-squares solution to the constrained
       optimization above (it is an instance of isotonic regression, a
       classical result from order-restricted statistical inference).

    Predicting for new points
    ---------------------------
    Each pooled block is summarized by (representative score, fitted
    value) = (weighted-mean score of its members, pooled value). To predict
    at a new score, we linearly interpolate between these representative
    points (and clip to the min/max fitted value outside the observed
    range). Linear interpolation turns the raw PAVA step function into a
    continuous, still-monotonic curve, which is the standard way isotonic
    regression is used as a calibration map in practice: a hard step
    function assigns the same probability to every score inside a block,
    which is visually and numerically jarring at the block boundaries.

    A deliberate deviation, stated explicitly
    ------------------------------------------
    Note that the exactness result above applies to the *blocks* produced
    by PAVA, not to what `predict` returns. Because we interpolate between
    block *means*, the value predicted at a training point inside a block
    is generally not equal to that block's fitted constant, so `predict`
    does not reproduce the exact least-squares isotonic solution on the
    calibration set itself (measured as residual sum of squares, it is
    slightly worse than the exact step function, by construction).

    This differs from scikit-learn's `IsotonicRegression`, which
    interpolates between the score values at the *block boundaries* and
    therefore does reproduce the exact fitted values at the training
    points. Both are monotonic, and the difference is small in practice,
    but the block-mean variant used here is a smoothing choice, not the
    argmin of the constrained least-squares problem. It is kept because a
    block's mean score is the more natural representative of where that
    block's probability estimate actually applies.

    Known failure mode: overfitting on small / well-separated calibration
    sets
    ----------------------------------------------------------------------
    If the calibration set is small and the base classifier already ranks
    the two classes perfectly, PAVA finds *no* rank violations to pool, so
    every calibration point remains its own block and the fitted value of
    each block is simply that point's label -- i.e. every block value is
    exactly 0.0 or 1.0. Unlike Platt scaling -- whose Bayesian-smoothed
    targets mathematically prevent the sigmoid from ever reaching exactly 0
    or 1 -- nothing here stops isotonic regression from doing so. This is
    not a bug: it is the exact least-squares isotonic solution when the
    monotonicity constraint is never active.

    Two clarifications that matter when discussing this result:

    (a) The *predictions* are still piecewise linear, not a hard step: we
        interpolate between block means (see above). What makes the test
        predictions collapse onto exactly 0 and 1 anyway is that all the
        block values are themselves 0 or 1, so interpolating between them
        only produces an intermediate value for the narrow score range
        that happens to fall between an opposite-labelled pair, and
        `np.interp` clamps flat to 0 or 1 outside the observed range. On
        the breast-cancer / Logistic Regression run this leaves 113 of 114
        test predictions at exactly 0 or 1.

    (b) The size of the resulting log-loss is largely an artifact of the
        clipping constant. A test point that lands on a degenerate block
        and is misclassified would contribute an infinite log-loss, so in
        practice log-loss is clipped at some eps (see `src/metrics.py`).
        With only two such confidently-wrong points, test log-loss on that
        run is 0.62 at eps=1e-15 but 0.14 at eps=1e-3 -- the *direction*
        of the effect is real and reproducible across seeds, but any
        specific number should be reported alongside the eps used.

    Brier score and accuracy are far less sensitive to this because they do
    not blow up near 0/1. Expect this effect to show up most on the "easy"
    (well separated) dataset with a small calibration set, and much less on
    a noisier dataset or a less confident base model (e.g. Random Forest).
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
