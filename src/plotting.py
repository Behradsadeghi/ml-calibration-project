"""
Reliability-diagram plotting utilities (matplotlib only).

A reliability diagram plots, for each bin of predicted probability, the
empirical fraction of positives observed in that bin against the mean
predicted probability in that bin. A perfectly calibrated model lies on the
y=x diagonal: "among all the times I said 80% confident, I was right about
80% of the time." Points ABOVE the diagonal mean the model is
under-confident in that bin (true frequency higher than predicted); points
BELOW the diagonal mean over-confident (true frequency lower than
predicted).

We follow Niculescu-Mizil & Caruana (2005) in pairing each curve with a
histogram of predicted probabilities underneath, since a reliability curve
computed from very few points in a bin is noisy and can be misleading on
its own -- the histogram shows how much to trust each part of the curve.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from .metrics import reliability_bins


def plot_reliability_curve(ax, y_true, y_prob, n_bins=10, label=None, marker="o"):
    """Draw a single reliability curve (predicted vs. observed frequency)
    onto an existing matplotlib Axes, plus the y=x diagonal if not already
    drawn. Returns the Line2D so callers can reuse its color."""
    bin_mean_pred, bin_frac_pos, bin_counts, _ = reliability_bins(y_true, y_prob, n_bins=n_bins)
    valid = bin_counts > 0
    line, = ax.plot(
        bin_mean_pred[valid], bin_frac_pos[valid],
        marker=marker, markersize=5, linewidth=1.5, label=label,
    )
    return line


def _ensure_diagonal(ax):
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfectly calibrated")


def plot_reliability_comparison(
    y_true, prob_dict, n_bins=10, title="", save_path=None,
):
    """
    Overlay several reliability curves (e.g. {"Uncalibrated": p0, "Platt":
    p1, "Isotonic": p2}) on one set of axes, with a shared histogram of
    predicted-probability counts underneath (one bar group per curve).

    Parameters
    ----------
    y_true : (n,) true binary labels (shared across all curves -- same test
             set, different probability estimates).
    prob_dict : dict[str, (n,) array] mapping a method name to its
                predicted probabilities on that same test set.
    n_bins : number of equal-width bins in [0, 1].
    save_path : if given, save the figure to this path (PNG, tight bbox).
    """
    fig, (ax_curve, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    _ensure_diagonal(ax_curve)
    colors = {}
    for name, p in prob_dict.items():
        line = plot_reliability_curve(ax_curve, y_true, p, n_bins=n_bins, label=name)
        colors[name] = line.get_color()

    ax_curve.set_ylabel("Observed frequency of positives")
    ax_curve.set_xlim(-0.02, 1.02)
    ax_curve.set_ylim(-0.02, 1.02)
    ax_curve.set_title(title)
    ax_curve.legend(loc="upper left", fontsize=9)
    ax_curve.grid(alpha=0.3)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    n_methods = len(prob_dict)
    for i, (name, p) in enumerate(prob_dict.items()):
        counts, _ = np.histogram(p, bins=bin_edges)
        offset = (i - (n_methods - 1) / 2) * (bin_width / (n_methods + 1))
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2 + offset
        ax_hist.bar(centers, counts, width=bin_width / (n_methods + 1), color=colors[name], label=name)

    ax_hist.set_xlabel("Mean predicted probability (bin)")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


_METHOD_ORDER = ["Uncalibrated", "Platt", "Isotonic"]
_METHOD_COLORS = {"Uncalibrated": "#1f77b4", "Platt": "#ff7f0e", "Isotonic": "#2ca02c"}


def plot_multiseed_stability(summary_df, metrics=("log_loss", "brier_score", "ece"), save_path=None):
    """
    Grid of bar charts (rows = metrics, columns = dataset x model
    combinations) showing, per method, the mean test-set metric across
    many random train/calibration/test splits with a +/- 1 std error bar.

    `summary_df` is the aggregated output of
    `src.experiment.run_multi_seed` (one row per dataset x model x method,
    with `{metric}_mean` / `{metric}_std` columns).

    Why this plot matters
    -----------------------
    A single reliability diagram or a single metrics table (as produced by
    `src.experiment.main`) is one sample from "what would happen with a
    different random split." A tall bar is a real effect; a tall bar with
    an error bar comparable to its own height is mostly noise from one
    particular split, and should not be over-interpreted (e.g. compared
    method-vs-method) without seeing the spread shown here.
    """
    combos = list(
        summary_df[["dataset", "model"]].drop_duplicates().itertuples(index=False, name=None)
    )
    n_rows, n_cols = len(metrics), len(combos)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.0 * n_rows), squeeze=False)

    for i, metric in enumerate(metrics):
        for j, (dataset, model) in enumerate(combos):
            ax = axes[i][j]
            sub = summary_df[(summary_df.dataset == dataset) & (summary_df.model == model)]
            sub = sub.set_index("method").reindex(_METHOD_ORDER)
            means = sub[f"{metric}_mean"].to_numpy(dtype=float)
            stds = sub[f"{metric}_std"].to_numpy(dtype=float)
            bar_colors = [_METHOD_COLORS[m] for m in _METHOD_ORDER]
            ax.bar(_METHOD_ORDER, means, yerr=stds, capsize=4, color=bar_colors)
            if i == 0:
                ax.set_title(f"{dataset}\n{model}", fontsize=10)
            if j == 0:
                ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=30)
            ax.grid(alpha=0.3, axis="y")

    n_seeds = int(summary_df["n_seeds"].iloc[0]) if "n_seeds" in summary_df else "?"
    fig.suptitle(f"Multi-seed stability across {n_seeds} random splits: mean ± std (test-set metrics)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
