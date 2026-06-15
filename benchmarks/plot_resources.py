import json
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import load_config

# Plot order matches plot_losses.py so the default color cycle assigns the
# same colors across both figures (C0=PyTorch, C1=Equinox, C2=nnzig).
_LIBS = ("pytorch", "equinox", "nnzig")
_LABELS = {"pytorch": "PyTorch", "equinox": "Equinox", "nnzig": "nnzig"}
# Libraries compared against nnzig in the ratio panels.
_RATIO_LIBS = ("pytorch", "equinox")
_REF = "nnzig"


def _mean_std(samples):
    """Return (mean, sample-std ddof=1) of a list; std is 0.0 for n=1."""
    arr = np.asarray(samples, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def _stats_arrays(samples, scale=1.0):
    """Per-N (means, stds) arrays from a list of sample lists."""
    means, stds = [], []
    for s in samples:
        m, sd = _mean_std(s)
        means.append(m * scale)
        stds.append(sd * scale)
    return np.array(means), np.array(stds)


def _draw_main(ax, data, n_arr, n_values, field, ylabel, title, scale=1.0):
    """Log-scale main panel: mean line + std band for every library."""
    means_by_lib = {}
    for lib in _LIBS:
        means, stds = _stats_arrays(data[lib][field], scale=scale)
        means_by_lib[lib] = means
        ax.plot(n_arr, means, label=_LABELS[lib], alpha=0.8)
        ax.fill_between(n_arr, means - stds, means + stds, alpha=0.2)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(n_values)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    return means_by_lib


def _draw_ratio(ax, means_by_lib, n_arr, n_values):
    """Linear ratio panel: (lib - nnzig) / nnzig * 100 for PyTorch/Equinox."""
    ref = means_by_lib[_REF]
    for lib in _RATIO_LIBS:
        pct = (means_by_lib[lib] - ref) / ref * 100.0
        ax.plot(n_arr, pct, label=_LABELS[lib], alpha=0.8,
                marker="o", markersize=3)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("N")
    ax.set_ylabel("vs {} (%)".format(_LABELS[_REF]))
    ax.set_xticks(n_values)
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    config = load_config()
    outputs = config["outputs"]

    print("[info] Loading resource data...")

    data = {}
    for lib in _LIBS:
        with open(outputs["resources_" + lib], "r") as f:
            data[lib] = json.load(f)

    n_values = data[_LIBS[0]]["N_values"]
    for lib in _LIBS[1:]:
        if data[lib]["N_values"] != n_values:
            raise ValueError(
                "N_values mismatch: {} has {} but {} has {}".format(
                    _LIBS[0], n_values, lib, data[lib]["N_values"]
                )
            )
    n_arr = np.array(n_values, dtype=float)

    # 2x2 grid: top row = main (log) panels, bottom row = ratio (linear)
    # panels at 1/3 height, sharing each column's x-axis with no vertical
    # gap (hspace=0). sharex="col" hides the x tick labels on the top row.
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 6.5), sharex="col",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0, "wspace": 0.15},
    )
    (ax_time, ax_mem), (ax_time_r, ax_mem_r) = axes

    time_means = _draw_main(
        ax_time, data, n_arr, n_values,
        "time_seconds", "Time (seconds)", "Training Time",
    )
    mem_means = _draw_main(
        ax_mem, data, n_arr, n_values,
        "max_rss_kbytes", "Memory (MB)", "Peak Memory", scale=1.0 / 1024.0,
    )

    _draw_ratio(ax_time_r, time_means, n_arr, n_values)
    _draw_ratio(ax_mem_r, mem_means, n_arr, n_values)

    output_file = outputs["resources_plot"]
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[info] Saved plot to '" + output_file + "'")
