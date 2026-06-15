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


def _mean_std(samples):
    """Return (mean, sample-std ddof=1) of a list; std is 0.0 for n=1."""
    arr = np.asarray(samples, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Time panel ---
    for lib in _LIBS:
        stats = [_mean_std(t) for t in data[lib]["time_seconds"]]
        means = np.array([m for m, _ in stats])
        stds = np.array([s for _, s in stats])
        axes[0].plot(n_arr, means, label=_LABELS[lib], alpha=0.8)
        axes[0].fill_between(n_arr, means - stds, means + stds, alpha=0.2)
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title("Training Time")
    axes[0].set_xticks(n_values)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Memory panel (kbytes -> MB) ---
    for lib in _LIBS:
        stats = [_mean_std(r) for r in data[lib]["max_rss_kbytes"]]
        means = np.array([m / 1024.0 for m, _ in stats])
        stds = np.array([s / 1024.0 for _, s in stats])
        axes[1].plot(n_arr, means, label=_LABELS[lib], alpha=0.8)
        axes[1].fill_between(n_arr, means - stds, means + stds, alpha=0.2)
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("Memory (MB)")
    axes[1].set_title("Peak Memory")
    axes[1].set_xticks(n_values)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    output_file = outputs["resources_plot"]
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print("[info] Saved plot to '" + output_file + "'")
