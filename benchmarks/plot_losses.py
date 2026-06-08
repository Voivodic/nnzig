import numpy as np
import struct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_losses_binary(filename):
    header_format = "<QQQQ"
    header_size = struct.calcsize(header_format)
    with open(filename, "rb") as f:
        header_bytes = f.read(header_size)
        precision_bits, n_epochs, _, _ = struct.unpack(header_format, header_bytes)
        dtype = np.float64 if precision_bits == 64 else np.float32
        data = np.frombuffer(f.read(), dtype=dtype)
        train_losses = data[:n_epochs]
        val_losses = data[n_epochs:]
        return train_losses, val_losses


if __name__ == "__main__":
    print("[info] Loading losses...")

    pt_train, pt_val = read_losses_binary("losses_pytorch.bin")
    zig_train, zig_val = read_losses_binary("losses_zig.bin")

    epochs = np.arange(1, len(pt_train) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, pt_train, label="PyTorch", alpha=0.8)
    axes[0].plot(epochs, zig_train, label="nnzig", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, pt_val, label="PyTorch", alpha=0.8)
    axes[1].plot(epochs, zig_val, label="nnzig", alpha=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    output_file = "losses_plot.png"
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"[info] Saved plot to '{output_file}'")
