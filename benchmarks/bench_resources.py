#!/usr/bin/env python3
"""Resource benchmark: sweep network size N and record wall-clock time and
peak RSS for nnzig (two configs), PyTorch, Equinox, and TensorFlow.

nnzig is measured twice per N to expose the memory/speed trade-off of its
compile-time ``batchSizeCompute`` parameter:

  * ``nnzig``        -- batchSizeCompute = batchSize (full-batch
                        vectorization; fast but more memory).
  * ``nnzig (bC=1)`` -- batchSizeCompute = 1 (one sample per op; slower
                        but less memory).

The Python libraries (PyTorch, Equinox, TensorFlow) are insensitive to
batchSizeCompute, so they are run once per N.

For each N in --n-values (default [2, 4, 8, 16, 32]) the network shape is
set to [N, 2N, 2N, N], the dataset is regenerated, nnzig is rebuilt (once
per batchSizeCompute config), and each library is run --reps times under
GNU ``time -v``. The two reported fields ("Elapsed (wall clock) time" and
"Maximum resident set size") are saved per library as JSON:

    {
      "N_values":       [2, 4, 8, 16, 32],
      "time_seconds":   [[rep1, rep2, ...], ...],   # aligned with N_values
      "max_rss_kbytes": [[rep1, rep2, ...], ...]
    }

plot_resources.py reads these files back. This script overwrites the JSON
on each run (no accumulation across invocations). It must be run from the
repository root. config.json/params.zon are backed up and restored in a
finally block.

Usage:
    python3 benchmarks/bench_resources.py [--reps N] [--n-values 2 4 8 ...]
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent
CONFIG = BENCH / "config.json"
PARAMS = BENCH / "params.zon"

sys.path.insert(0, str(BENCH))
from config import load_config, write_params_zon  # noqa: E402

DEFAULT_N_VALUES = [2, 4, 8, 16, 32]
DEFAULT_REPS = 3


_DEFAULT_DESC = "Sweep network size N and record time/memory for nnzig (2 configs), PyTorch, Equinox, TensorFlow."


def parse_args():
    parser = argparse.ArgumentParser(description=_DEFAULT_DESC)
    parser.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPS,
        help=f"repetitions per N value (default {DEFAULT_REPS})",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=DEFAULT_N_VALUES,
        help=f"N values to sweep (default {DEFAULT_N_VALUES})",
    )
    return parser.parse_args()


def run(cmd, cwd=None, check=True):
    """Run cmd, streaming the last few lines of combined output. Returns
    the CompletedProcess (stdout is combined stdout+stderr)."""
    print("  $ " + " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    tail = "\n".join(proc.stdout.splitlines()[-6:])
    if tail:
        print("    " + tail.replace("\n", "\n    "))
    if check and proc.returncode != 0:
        raise RuntimeError("command failed: " + " ".join(cmd))
    return proc


def resolve_nix_app(app_name):
    """Return the store path of the app's wrapper script (its 'program').

    Uses separate stdout/stderr capture (not the ``run`` helper, which
    merges them) so nix's "dirty git tree" warning doesn't get concatenated
    into the returned path."""
    proc = subprocess.run(
        [
            "nix",
            "eval",
            "--impure",
            "--raw",
            f"./benchmarks#apps.x86_64-linux.{app_name}.program",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nix eval failed for '{app_name}': {proc.stderr}")
    return proc.stdout.strip()


def ensure_app_realized(app_name, program_path):
    """GNU time can only wrap a path that already exists, so if the resolved
    program path isn't in the store yet, do one warm-up ``nix run`` to build
    the env. Its side effects on the losses files are harmless."""
    if os.path.exists(program_path):
        return
    print(f"[bench] building nix env for '{app_name}' ...")
    run(
        [
            "nix",
            "run",
            "--no-pure-eval",
            "--impure",
            f"./benchmarks#{app_name}",
        ],
        cwd=str(REPO),
    )


def parse_wall_clock(value):
    """Parse GNU time's 'm:ss' or 'h:mm:ss' into seconds (float)."""
    parts = value.split(":")
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    raise ValueError("unexpected wall-clock format: " + value)


def parse_time_output(text):
    """Extract (wall_seconds, max_rss_kbytes) from GNU `time -v` output."""
    wall = None
    rss = None
    for line in text.splitlines():
        stripped = line.strip()
        # Each label ends with "(<units>): <value>"; the value itself may
        # contain colons (e.g. "0:00.05"), so split on the ") " that closes
        # the units hint rather than on the last colon.
        if stripped.startswith("Elapsed (wall clock) time"):
            value = stripped.split("): ", 1)[-1].strip()
            wall = parse_wall_clock(value)
        elif stripped.startswith("Maximum resident set size"):
            value = stripped.split("): ", 1)[-1].strip()
            rss = int(value)
    if wall is None or rss is None:
        raise RuntimeError("could not parse `time -v` output:\n" + text)
    return wall, rss


def run_timed(cmd, cwd):
    """Run cmd under `time -v`, return (wall_seconds, max_rss_kbytes).

    GNU time writes its report to a temp file (-o) so it doesn't mix with
    the command's own stderr; the command's stdout/stderr are still captured
    in proc for error reporting."""
    fd, time_file = tempfile.mkstemp(suffix=".time")
    os.close(fd)
    try:
        proc = subprocess.run(
            ["time", "-v", "-o", time_file] + list(cmd),
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"timed command failed (rc={proc.returncode}): "
                f"{' '.join(cmd)}\n--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}"
            )
        with open(time_file) as f:
            time_output = f.read()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(time_file)
    return parse_time_output(time_output)


def set_n_neurons(N):
    """Point network.nNeurons at [N, 2N, 2N, N] and regenerate the dataset +
    params.zon. generate-data reads the live config.json, then writes both
    dataset_benchmark.bin (new dim_in = N) and params.zon (new nNeurons)."""
    cfg = load_config(CONFIG, validate=False)
    cfg["network"]["nNeurons"] = [N, 2 * N, 2 * N, N]
    with open(CONFIG, "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")
    run(
        ["nix", "run", "--no-pure-eval", "--impure", "./benchmarks#generate-data"],
        cwd=str(REPO),
    )


def set_batch_size_compute(value):
    """Set training.batchSizeCompute in config.json and regenerate
    params.zon WITHOUT regenerating the dataset. Because nnzig reads
    batchSizeCompute at compile time, the caller must ``zig build`` after
    this for the change to take effect in the binary."""
    cfg = load_config(CONFIG, validate=False)
    cfg["training"]["batchSizeCompute"] = value
    with open(CONFIG, "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")
    write_params_zon(cfg, PARAMS)


def main():
    args = parse_args()
    n_values = args.n_values
    reps = args.reps

    if shutil.which("time") is None:
        raise RuntimeError("GNU `time` not found on PATH (needed for `time -v`).")

    backup_cfg = CONFIG.read_text()
    backup_params = PARAMS.read_text()

    outputs = load_config(CONFIG, validate=False)["outputs"]

    # Capture the original batchSize once: it is the full-batch value we
    # restore for the "nnzig" config. batchSizeCompute is compile-time, so
    # each nnzig config requires its own rebuild.
    batch_size_value = load_config(CONFIG, validate=False)["training"]["batchSize"]

    try:
        # Ensure output directories exist (the nnzig binary can't create
        # benchmarks/losses/ itself; the Python scripts do their own
        # makedirs, but this covers all of them upfront).
        for sub in ("losses", "plots", "resources"):
            (BENCH / sub).mkdir(parents=True, exist_ok=True)

        # Resolve pytorch/equinox/tensorflow app paths once so per-rep
        # invocations skip nix's flake-eval overhead (~100-500ms each).
        print("[bench] resolving nix app paths ...")
        pytorch_app = resolve_nix_app("run-pytorch")
        equinox_app = resolve_nix_app("run-equinox")
        tensorflow_app = resolve_nix_app("run-tensorflow")
        ensure_app_realized("run-pytorch", pytorch_app)
        ensure_app_realized("run-equinox", equinox_app)
        ensure_app_realized("run-tensorflow", tensorflow_app)

        # results[lib][N] = [(wall, rss), ...]
        results = {
            lib: {}
            for lib in (
                "nnzig",
                "nnzig_b1",
                "pytorch",
                "equinox",
                "tensorflow",
            )
        }

        for N in n_values:
            print(f"\n[bench] ===== N = {N} (shape [{N}, {2 * N}, {2 * N}, {N}]) =====")
            # set_n_neurons regenerates the dataset + params.zon (leaving
            # batchSizeCompute at its current value); do it first, then
            # override batchSizeCompute per nnzig config below.
            set_n_neurons(N)

            # --- nnzig full-batch (batchSizeCompute = batchSize) ---
            set_batch_size_compute(batch_size_value)
            print(f"[bench] building nnzig (batchSizeCompute={batch_size_value}) ...")
            run(["zig", "build"], cwd=str(REPO))
            for r in range(reps):
                print(f"[bench]   rep {r + 1}/{reps} [nnzig]", flush=True)
                wall, rss = run_timed(["zig-out/bin/benchmark"], cwd=str(REPO))
                results["nnzig"].setdefault(N, []).append((wall, rss))
                print(f"[bench]     {'nnzig':<13}: {wall:.3f}s, {rss} KB")

            # --- nnzig sequential (batchSizeCompute = 1) ---
            set_batch_size_compute(1)
            print("[bench] building nnzig (batchSizeCompute=1) ...")
            run(["zig", "build"], cwd=str(REPO))
            for r in range(reps):
                print(f"[bench]   rep {r + 1}/{reps} [nnzig_b1]", flush=True)
                wall, rss = run_timed(["zig-out/bin/benchmark"], cwd=str(REPO))
                results["nnzig_b1"].setdefault(N, []).append((wall, rss))
                print(f"[bench]     {'nnzig (bC=1)':<13}: {wall:.3f}s, {rss} KB")

            # --- Python libraries (unaffected by batchSizeCompute) ---
            for r in range(reps):
                print(f"[bench]   rep {r + 1}/{reps} [python]", flush=True)

                wall, rss = run_timed([pytorch_app], cwd=str(REPO))
                results["pytorch"].setdefault(N, []).append((wall, rss))
                print(f"[bench]     {'pytorch':<13}: {wall:.3f}s, {rss} KB")

                wall, rss = run_timed([equinox_app], cwd=str(REPO))
                results["equinox"].setdefault(N, []).append((wall, rss))
                print(f"[bench]     {'equinox':<13}: {wall:.3f}s, {rss} KB")

                wall, rss = run_timed([tensorflow_app], cwd=str(REPO))
                results["tensorflow"].setdefault(N, []).append((wall, rss))
                print(f"[bench]     {'tensorflow':<13}: {wall:.3f}s, {rss} KB")

        # Persist per-library JSON (overwrite mode). The output keys are
        # "resources_<lib>", which lines up with all five lib keys.
        for lib in ("nnzig", "nnzig_b1", "pytorch", "equinox", "tensorflow"):
            data = {
                "N_values": list(n_values),
                "time_seconds": [[t for t, _ in results[lib][N]] for N in n_values],
                "max_rss_kbytes": [[r for _, r in results[lib][N]] for N in n_values],
            }
            out = BENCH / outputs["resources_" + lib]
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[bench] wrote {out.name}")
    finally:
        CONFIG.write_text(backup_cfg)
        PARAMS.write_text(backup_params)
        print("\n[bench] restored config.json and params.zon")


if __name__ == "__main__":
    main()
