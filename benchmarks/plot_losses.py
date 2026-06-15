import numpy as np
import os
import struct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import load_config

# Precision in *bytes* (the first header field, matching @sizeOf(T) in
# src/io/binary.zig) -> numpy dtype used to decode the loss arrays.
_DTYPE_FROM_BYTES = {
    2: np.float16,
    4: np.float32,
    8: np.float64,
}


def read_losses_binary(filename):
    """Read a loss-curve file written by nnzig's saveLosses() or by
    run_pytorch.write_losses_binary().

    Layout: a 32-byte DataHeader (<QQQQ>) = [precision_bytes, n_epochs,
    dim_in=1, dim_out=1], followed by n_epochs train values then n_epochs
    validation values, each encoded at the file's precision.
    """
    header_format = "<QQQQ"
    header_size = struct.calcsize(header_format)
    with open(filename, "rb") as f:
        header_bytes = f.read(header_size)
        precision_bytes, n_epochs, _, _ = struct.unpack(header_format, header_bytes)
        if precision_bytes not in _DTYPE_FROM_BYTES:
            raise ValueError(
                "Unknown precision ({} bytes) in {}".format(precision_bytes, filename)
            )
        dtype = _DTYPE_FROM_BYTES[precision_bytes]
        data = np.frombuffer(f.read(), dtype=dtype)
        train_losses = data[:n_epochs]
        val_losses = data[n_epochs:2 * n_epochs]
        return train_losses, val_losses


if __name__ == "__main__":
    config = load_config()
    outputs = config["outputs"]

    print("[info] Loading losses...")

    pt_train, pt_val = read_losses_binary(outputs["losses_pytorch"])
    eq_train, eq_val = read_losses_binary(outputs["losses_equinox"])
    zig_train, zig_val = read_losses_binary(outputs["losses_zig"])

    if not (len(pt_train) == len(eq_train) == len(zig_train)
            == len(pt_val) == len(eq_val) == len(zig_val)):
        raise ValueError(
            "Loss arrays have mismatched lengths: pytorch train/val = {}/{}, "
            "equinox train/val = {}/{}, zig train/val = {}/{}. All runs must "
            "use the same nEpochs.".format(
                len(pt_train), len(pt_val),
                len(eq_train), len(eq_val),
                len(zig_train), len(zig_val),
            )
        )

    epochs = np.arange(1, len(pt_train) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, pt_train, label="PyTorch", alpha=0.8)
    axes[0].plot(epochs, eq_train, label="Equinox", alpha=0.8)
    axes[0].plot(epochs, zig_train, label="nnzig", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, pt_val, label="PyTorch", alpha=0.8)
    axes[1].plot(epochs, eq_val, label="Equinox", alpha=0.8)
    axes[1].plot(epochs, zig_val, label="nnzig", alpha=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].set_title("Validation Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    output_file = outputs["losses_plot"]
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print("[info] Saved plot to '" + output_file + "'")
