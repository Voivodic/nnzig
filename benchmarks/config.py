"""Shared benchmark configuration helpers.

All benchmark parameters live in ``config.json``. This module loads them,
validates them (mirroring the compile-time checks in ``src/core/params.zig``
so config mistakes surface here instead of as Zig build errors), and provides
the ZON writer used to keep ``params.zon`` in sync with the JSON config.

It only depends on the standard library so it can be imported by the
numpy/torch/matplotlib scripts without pulling in heavy dependencies.
"""

import json
from pathlib import Path

# Resolve relative to the current working directory (not __file__). The scripts
# are designed to run from the benchmarks/ directory; the nix flake apps cd
# there automatically. Using __file__ would point into the read-only nix store
# when run via `nix run`, breaking writes to params.zon.
CONFIG_PATH = Path("config.json")
PARAMS_PATH = Path("params.zon")

# Field order must match the canonical benchmarks/params.zon layout. Each entry
# is (zon_field_name, config_path_tuple).
_PARAMS_FIELDS = [
    ("precision", ("network", "precision")),
    ("numThreads", ("network", "numThreads")),
    ("nNeurons", ("network", "nNeurons")),
    ("activations", ("network", "activations")),
    ("lossFunc", ("network", "lossFunc")),
    ("normalization", ("network", "normalization")),
    ("seed", ("training", "seed")),
    ("rTrain", ("training", "rTrain")),
    ("rVal", ("training", "rVal")),
    ("eps", ("training", "eps")),
    ("beta1", ("training", "beta1")),
    ("beta2", ("training", "beta2")),
    ("lr", ("training", "lr")),
    ("nEpochs", ("training", "nEpochs")),
    ("batchSize", ("training", "batchSize")),
    ("printEvery", ("training", "printEvery")),
]

# Values permitted by the enums in src/core/params.zig. Keep these in sync.
_VALID_PRECISIONS = (16, 32, 64)
_VALID_LOSSES = ("MSE",)
_VALID_NORMALIZATIONS = ("meanStd",)
_VALID_ACTIVATIONS = ("none", "relu", "tanh", "sigmoid")


def load_config(path=CONFIG_PATH, validate=True):
    with open(path, "r") as f:
        config = json.load(f)
    if validate:
        validate_config(config)
    return config


def _get(config, path_tuple):
    value = config
    for key in path_tuple:
        value = value[key]
    return value


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config):
    """Validate the config against the same constraints enforced at compile
    time in src/core/params.zig. Raises ValueError on the first problem."""
    network = config["network"]
    training = config["training"]
    dataset = config["dataset"]
    outputs = config["outputs"]

    # --- Network shape ---
    n_neurons = network["nNeurons"]
    _check(
        isinstance(n_neurons, list) and len(n_neurons) >= 2,
        "network.nNeurons must be a list with at least two elements "
        "(input and output layers).",
    )
    _check(
        all(isinstance(n, int) and n > 0 for n in n_neurons),
        "network.nNeurons entries must be positive integers.",
    )

    activations = network["activations"]
    n_hidden = len(n_neurons) - 1
    _check(
        len(activations) == n_hidden,
        "network.activations must have exactly len(nNeurons) - 1 = {} "
        "entries, got {}.".format(n_hidden, len(activations)),
    )
    bad = [a for a in activations if a not in _VALID_ACTIVATIONS]
    _check(not bad, "Invalid activation(s): {}. Valid: {}.".format(bad, _VALID_ACTIVATIONS))

    _check(
        network["precision"] in _VALID_PRECISIONS,
        "network.precision must be one of {}.".format(_VALID_PRECISIONS),
    )
    _check(
        network["lossFunc"] in _VALID_LOSSES,
        "network.lossFunc must be one of {}.".format(_VALID_LOSSES),
    )
    _check(
        network["normalization"] in _VALID_NORMALIZATIONS,
        "network.normalization must be one of {}.".format(_VALID_NORMALIZATIONS),
    )
    _check(
        isinstance(network["numThreads"], int) and network["numThreads"] >= 1,
        "network.numThreads must be a positive integer.",
    )

    # --- Dataset ---
    _check(
        isinstance(dataset["num_points"], int) and dataset["num_points"] > 0,
        "dataset.num_points must be a positive integer.",
    )
    _check(dataset["file"], "dataset.file must be a non-empty string.")
    _check(
        isinstance(dataset.get("seed"), int) and dataset["seed"] >= 0,
        "dataset.seed must be a non-negative integer (controls only the "
        "data generation RNG, independent of training.seed).",
    )

    # --- Training splits ---
    r_train = training["rTrain"]
    r_val = training["rVal"]
    _check(0.0 <= r_train <= 1.0, "training.rTrain must be in [0.0, 1.0].")
    _check(0.0 <= r_val <= 1.0, "training.rVal must be in [0.0, 1.0].")
    _check(
        r_train + r_val <= 1.0,
        "training.rTrain + training.rVal must be <= 1.0 (the remainder is "
        "the test split).",
    )

    # --- Adam / optimizer hyperparameters ---
    _check(training["eps"] > 0.0, "training.eps must be > 0.0.")
    _check(0.0 <= training["beta1"] <= 1.0, "training.beta1 must be in [0.0, 1.0].")
    _check(0.0 <= training["beta2"] <= 1.0, "training.beta2 must be in [0.0, 1.0].")
    _check(training["lr"] > 0.0, "training.lr must be > 0.0.")

    _check(
        isinstance(training["nEpochs"], int) and training["nEpochs"] >= 1,
        "training.nEpochs must be a positive integer.",
    )
    _check(
        isinstance(training["batchSize"], int) and training["batchSize"] >= 1,
        "training.batchSize must be a positive integer.",
    )
    _check(
        isinstance(training["printEvery"], int) and training["printEvery"] >= 0,
        "training.printEvery must be a non-negative integer.",
    )
    _check(
        isinstance(training["seed"], int) and training["seed"] >= 0,
        "training.seed must be a non-negative integer.",
    )

    # --- Sanity: batch size vs. training set ---
    n_train = int(dataset["num_points"] * r_train)
    _check(
        n_train >= 1,
        "dataset.num_points * training.rTrain must yield at least 1 training "
        "sample (currently {}).".format(n_train),
    )

    # --- Outputs ---
    for key in ("losses_zig", "losses_pytorch", "losses_equinox", "losses_plot"):
        _check(outputs.get(key), "outputs.{} must be a non-empty string.".format(key))


def _format_zon_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return '"' + value + '"'
    if isinstance(value, list):
        inner = ", ".join(_format_zon_value(v) for v in value)
        return ".{ " + inner + " }"
    raise TypeError("Cannot format value of type " + type(value).__name__)


def render_params_zon(config):
    """Return the contents of params.zon (no comments) for the given config."""
    lines = [
        "    ." + name + " = " + _format_zon_value(_get(config, path)) + ","
        for name, path in _PARAMS_FIELDS
    ]
    return ".{\n" + "\n".join(lines) + "\n}\n"


def write_params_zon(config, path=PARAMS_PATH):
    """Write benchmarks/params.zon from the config."""
    validate_config(config)
    with open(path, "w") as f:
        f.write(render_params_zon(config))
    print("Successfully wrote params to '" + str(path) + "'.")
