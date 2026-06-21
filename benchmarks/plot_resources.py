import json
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import load_config

# Plot order matches plot_losses.py so the default color cycle assigns the
# same colors across both figures
# (C0=PyTorch, C1=Equinox, C2=TensorFlow, C3=nnzig, C4=nnzig(bC=1)).
_LIBS = ("pytorch", "equinox", "tensorflow", "nnzig_b1", "nnzig", )
_LABELS = {
    "pytorch": "PyTorch",
    "equinox": "Equinox",
    "tensorflow": "TensorFlow",
    "nnzig_b1": "nnzig (bC=1)",
    "nnzig": "nnzig",
}
# Libraries compared against nnzig (full-batch) in the ratio panels: every
# other series, including the bC=1 variant, to show the memory/speed gap.
_RATIO_LIBS = ("pytorch", "equinox", "tensorflow", "nnzig_b1")
_REF = "nnzig"


def _n_neurons(N):
    """Layer sizes for sweep value N. Matches the [N, 2N, 2N, N] shape set by
    bench_resources.set_n_neurons (the resource sweep always uses 4 layers)."""
    return [N, 2 * N, 2 * N, N]


def _num_params(N):
    """Total trainable parameters (weights + biases) of the [N, 2N, 2N, N]
    MLP. Used as the x-axis so the horizontal axis measures model size rather
    than the raw sweep index N."""
    neurons = _n_neurons(N)
    return sum(
        inp * out + out for inp, out in zip(neurons[:-1], neurons[1:], strict=True)
    )


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


def _draw_main(ax, data, x_arr, field, ylabel, title, scale=1.0):
    """Log-log main panel: mean line + std band for every library. Both axes
    are log-scaled; the x-axis is the total parameter count (x_arr /
    x_values), not N."""
    means_by_lib = {}
    for lib in _LIBS:
        means, stds = _stats_arrays(data[lib][field], scale=scale)
        means_by_lib[lib] = means
        ax.plot(x_arr, means, label=_LABELS[lib], alpha=0.8)
        ax.fill_between(x_arr, means - stds, means + stds, alpha=0.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    return means_by_lib


def _draw_ratio(ax, means_by_lib, x_arr):
    """Ratio panel: (lib - ref) / ref * 100 for every lib in _RATIO_LIBS
    (PyTorch, Equinox, TensorFlow, nnzig bC=1), vs nnzig. The y-axis is
    symmetric-log so large positive (slower/bigger) and small negative
    (faster/smaller) deviations are both visible; the x-axis is log-scaled
    and shows the total parameter count."""
    ref = means_by_lib[_REF]
    for lib in _RATIO_LIBS:
        pct = (means_by_lib[lib] - ref) / ref
        ax.plot(x_arr, pct, label=_LABELS[lib], alpha=0.8, marker="o", markersize=3)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Total parameters")
    ax.set_xscale("log")
    ax.set_ylabel(f"vs {_LABELS[_REF]} (Ratio-1)")
    ax.set_yscale("symlog")
    ax.grid(True, alpha=0.3, which="both")


if __name__ == "__main__":
    config = load_config()
    outputs = config["outputs"]

    print("[info] Loading resource data...")

    data = {}
    for lib in _LIBS:
        with open(outputs["resources_" + lib]) as f:
            data[lib] = json.load(f)

    n_values = data[_LIBS[0]]["N_values"]
    for lib in _LIBS[1:]:
        if data[lib]["N_values"] != n_values:
            raise ValueError(
                f"N_values mismatch: {_LIBS[0]} has {n_values} but {lib} "
                f"has {data[lib]['N_values']}"
            )
    # Plot against the total parameter count (weights + biases of the
    # [N, 2N, 2N, N] MLP) rather than the raw sweep index N, so the x-axis
    # measures model size.
    param_values = [_num_params(N) for N in n_values]
    x_arr = np.array(param_values, dtype=float)

    # 2x2 grid: top row = main (log) panels, bottom row = ratio (symlog)
    # panels at 1/3 height, sharing each column's x-axis with no vertical
    # gap (hspace=0). sharex="col" hides the x tick labels on the top row.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 6.5),
        sharex="col",
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.0, "wspace": 0.15},
    )
    (ax_time, ax_mem), (ax_time_r, ax_mem_r) = axes

    time_means = _draw_main(
        ax_time,
        data,
        x_arr,
        "time_seconds",
        "Time (seconds)",
        "Training Time",
    )
    mem_means = _draw_main(
        ax_mem,
        data,
        x_arr,
        "max_rss_kbytes",
        "Memory (MB)",
        "Peak Memory",
        scale=1.0 / 1024.0,
    )

    _draw_ratio(ax_time_r, time_means, x_arr)
    _draw_ratio(ax_mem_r, mem_means, x_arr)

    output_file = outputs["resources_plot"]
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[info] Saved plot to '" + output_file + "'")
