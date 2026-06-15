#!/usr/bin/env python3
"""Multi-seed benchmark to separate optimizer RNG from the dataset.

The dataset is generated ONCE (using ``dataset.seed``) and kept fixed. Then
both nnzig and PyTorch are trained for several different ``training.seed``
values (which control weight init + mini-batch shuffling). If nnzig's final
loss is sometimes below and sometimes above PyTorch's, the divergence seen at
a single seed is RNG-driven; if nnzig is consistently worse across all seeds,
that points to a real difference in the implementation.

This script only needs the Python standard library. It shells out to
``nix run ./benchmarks#...`` for the numpy/torch steps and ``zig build
benchmark`` for the nnzig step, so it must be run from the repository root.

Usage:
    python3 benchmarks/multi_seed.py [seed1 seed2 ...]

Seeds default to: 1 42 123 777 31337
"""

import array
import json
import statistics
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent
CONFIG = BENCH / "config.json"
PARAMS = BENCH / "params.zon"

sys.path.insert(0, str(BENCH))
from config import load_config, write_params_zon  # noqa: E402

DEFAULT_SEEDS = [1, 42, 123, 777, 31337]


def read_losses(path):
    """Read a loss-curve file ([precision_bytes, n_epochs, 1, 1] + train + val)."""
    with open(path, "rb") as f:
        pb, n_epochs, _, _ = struct.unpack("<QQQQ", f.read(32))
        raw = f.read()
    code = {2: "e", 4: "f", 8: "d"}[pb]
    train = array.array(code)
    train.frombytes(raw[: n_epochs * pb])
    val = array.array(code)
    val.frombytes(raw[n_epochs * pb : 2 * n_epochs * pb])
    return list(train), list(val)


def run(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Print the tail so failures are visible without flooding the log.
    tail = "\n".join(proc.stdout.splitlines()[-6:])
    if tail:
        print("    " + tail.replace("\n", "\n    "))
    if proc.returncode != 0:
        raise RuntimeError("command failed: " + " ".join(cmd))
    return proc.stdout


def set_training_seed(seed):
    """Point training.seed at `seed` and regenerate params.zon (nnzig reads it
    at compile time). The dataset is left untouched."""
    cfg = load_config(CONFIG, validate=False)
    cfg["training"]["seed"] = seed
    with open(CONFIG, "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")
    write_params_zon(cfg, PARAMS)


def main():
    seeds = [int(x) for x in sys.argv[1:]] or DEFAULT_SEEDS

    backup_cfg = CONFIG.read_text()
    backup_params = PARAMS.read_text()

    try:
        # 1. Generate the dataset ONCE with the configured dataset.seed.
        print("[multi-seed] Generating dataset once (dataset.seed) ...")
        run(["nix", "run", "./benchmarks#generate-data", "--no-pure-eval"])

        results = []
        for seed in seeds:
            print("\n[multi-seed] ===== training seed {} =====".format(seed))
            set_training_seed(seed)

            print("[multi-seed] nnzig (zig build benchmark) ...")
            run(["zig", "build", "benchmark", "--summary", "all"])
            z_tr, z_va = read_losses(BENCH / "losses" / "losses_zig.bin")

            print("[multi-seed] pytorch (nix run) ...")
            run(["nix", "run", "./benchmarks#run-pytorch", "--no-pure-eval"])
            p_tr, p_va = read_losses(BENCH / "losses" / "losses_pytorch.bin")

            results.append((seed, z_tr[-1], p_tr[-1], z_va[-1], p_va[-1]))

        # 2. Report.
        print("\n" + "=" * 74)
        print("{:>6} | {:>10} {:>10} | {:>10} {:>10} | {}".format(
            "seed", "zig_train", "pt_train", "zig_val", "pt_val", "lower-train"))
        print("-" * 74)
        zig_train_wins = 0
        zig_val_wins = 0
        train_ratios = []
        val_ratios = []
        for seed, zt, pt, zv, pv in results:
            wt = "zig" if zt < pt else "pytorch"
            wv = "zig" if zv < pv else "pytorch"
            zig_train_wins += zt < pt
            zig_val_wins += zv < pv
            train_ratios.append(zt / pt if pt else float("inf"))
            val_ratios.append(zv / pv if pv else float("inf"))
            print("{:6d} | {:10.6f} {:10.6f} | {:10.6f} {:10.6f} | tr={} val={}".format(
                seed, zt, pt, zv, pv, wt, wv))
        print("-" * 74)
        n = len(results)
        print("nnzig lower train loss: {}/{} seeds".format(zig_train_wins, n))
        print("nnzig lower val   loss: {}/{} seeds".format(zig_val_wins, n))
        print("zig/pt final train ratio: min={:.3f} med={:.3f} max={:.3f}".format(
            min(train_ratios), statistics.median(train_ratios), max(train_ratios)))
        print("zig/pt final val   ratio: min={:.3f} med={:.3f} max={:.3f}".format(
            min(val_ratios), statistics.median(val_ratios), max(val_ratios)))
        print()
        if min(train_ratios) < 1.0 and max(train_ratios) > 1.0:
            print("=> Ordering FLIPS across seeds: divergence is RNG-driven.")
        elif all(r > 1.0 for r in train_ratios):
            print("=> nnzig is consistently WORSE on train across all seeds: "
                  "likely a real implementation difference.")
        else:
            print("=> nnzig is consistently better on train: check the other side.")
    finally:
        CONFIG.write_text(backup_cfg)
        PARAMS.write_text(backup_params)
        print("\n[multi-seed] Restored original config.json and params.zon.")


if __name__ == "__main__":
    main()
