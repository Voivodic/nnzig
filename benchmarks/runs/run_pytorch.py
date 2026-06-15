import numpy as np
import os
import struct
import torch

from config import load_config

# Precision (in bits) -> (numpy dtype, torch dtype).
_PRECISION = {
    16: (np.float16, torch.float16),
    32: (np.float32, torch.float32),
    64: (np.float64, torch.float64),
}

_LOSS_FUNCS = {
    "MSE": torch.nn.MSELoss,
}


def read_numpy_binary(filename, precision_bits):
    """Read the benchmark dataset written by generate_data.py.

    The header's first field is the precision in *bytes* (matching the
    DataHeader convention in src/io/binary.zig, which stores @sizeOf(T)).
    """
    if precision_bits not in _PRECISION:
        raise ValueError("Invalid precision: " + str(precision_bits))
    np_dtype, _ = _PRECISION[precision_bits]

    # '<QQQQ' means Little-Endian, 4x Unsigned 64-bit integers
    header_format = "<QQQQ"
    header_size = struct.calcsize(header_format)

    with open(filename, "rb") as f:
        header_bytes = f.read(header_size)
        _, num_points, dim_in, dim_out = struct.unpack(header_format, header_bytes)

        data = np.frombuffer(f.read(), dtype=np_dtype)

        input_size = num_points * dim_in
        input_data = data[:input_size].reshape((num_points, dim_in)).copy()
        output_data = data[input_size:].reshape((num_points, dim_out)).copy()

        return input_data, output_data


def write_losses_binary(filename, train_losses, val_losses, precision_bits):
    """Write the training/validation loss curves.

    The on-disk layout matches nnzig's saveLosses() -> io.saveData() with
    dimIn=dimOut=1: a 32-byte DataHeader followed by the train then val
    arrays. The first header field is the precision in *bytes* (@sizeOf(T)),
    the second is the number of epochs, the last two are the (unused) dims.
    """
    if precision_bits not in _PRECISION:
        raise ValueError("Invalid precision: " + str(precision_bits))
    np_dtype, _ = _PRECISION[precision_bits]
    precision_bytes = precision_bits // 8

    header_format = "<QQQQ"
    header_bytes = struct.pack(
        header_format, precision_bytes, len(train_losses), 1, 1
    )
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "wb") as f:
        f.write(header_bytes)
        f.write(train_losses.astype(np_dtype).tobytes())
        f.write(val_losses.astype(np_dtype).tobytes())


class MLP(torch.nn.Module):
    def __init__(self, layer_sizes, activations):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if activations[i] == "relu":
                layers.append(torch.nn.ReLU())
            elif activations[i] == "sigmoid":
                layers.append(torch.nn.Sigmoid())
            elif activations[i] == "tanh":
                layers.append(torch.nn.Tanh())
            elif activations[i] == "none":
                pass
            else:
                raise ValueError("Unsupported activation: " + activations[i])
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    config = load_config()
    network = config["network"]
    training = config["training"]

    precision_bits = network["precision"]
    np_dtype, torch_dtype = _PRECISION[precision_bits]

    # Run all of PyTorch in the configured precision. This makes every float
    # tensor PyTorch creates internally (model parameters, activations,
    # optimizer state, ...) default to torch_dtype, so nothing silently falls
    # back to f64. The explicit .to(torch_dtype) and dtype=torch_dtype below
    # then match this default.
    torch.set_default_dtype(torch_dtype)

    print("[info] Initializing NN...")

    seed = training["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("[info] Loading dataset...")
    X, Y = read_numpy_binary(config["dataset"]["file"], precision_bits)

    print("[info] Computing normalization...")
    # Match nnzig's meanStd normalization exactly (src/eigen/normalizations.cpp):
    # population mean/std (ddof=0), x' = (x - mean) / std, with NO eps in the
    # denominator. The configured eps is only used by the Adam optimizer.
    norm = network["normalization"]
    if norm == "meanStd":
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    else:
        raise ValueError("Unsupported normalization: " + norm)

    N = X.shape[0]
    n_train = int(N * training["rTrain"])
    n_val = int(N * training["rVal"])

    X_t = torch.tensor(X[:n_train], dtype=torch_dtype)
    Y_t = torch.tensor(Y[:n_train], dtype=torch_dtype)
    X_v = torch.tensor(X[n_train:n_train + n_val], dtype=torch_dtype)
    Y_v = torch.tensor(Y[n_train:n_train + n_val], dtype=torch_dtype)
    del X, Y

    print("[info] Training network...")

    model = MLP(network["nNeurons"], network["activations"]).to(torch_dtype)
    # Match nnzig's init: every weight and bias drawn from N(0, 1).
    for p in model.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=1.0)

    loss_name = network["lossFunc"]
    if loss_name not in _LOSS_FUNCS:
        raise ValueError("Unsupported loss function: " + loss_name)
    # reduction="mean" divides the squared error by (batch * nOut), so the
    # gradient scale matches nnzig's (whose backProp divides by batchSize and
    # whose MSE kernel divides by nOut). Using "sum" here would leave PyTorch's
    # gradients ~batchSize*dim_out larger; Adam is scale-invariant except for
    # the eps term, so that mismatch would make nnzig's effective eps ~100x
    # larger and stall it at higher loss late in training.
    criterion = _LOSS_FUNCS[loss_name](reduction="mean")
    # NOTE: torch.optim.Adam uses a global step counter (incremented every
    # optimizer.step()). nnzig instead uses a per-epoch counter t = epoch+1 for
    # every batch within an epoch. This is a known algorithmic difference; both
    # use the same lr/beta1/beta2/eps.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training["lr"],
        betas=(training["beta1"], training["beta2"]),
        eps=training["eps"],
    )

    n_epochs = training["nEpochs"]
    batch_size = training["batchSize"]
    n_batches = n_train // batch_size
    n_remainder = n_train % batch_size
    train_losses = np.empty(n_epochs, dtype=np_dtype)
    val_losses = np.empty(n_epochs, dtype=np_dtype)

    for epoch in range(n_epochs):
        model.train()

        # Match nnzig: shuffle only the training indices each epoch so the
        # validation set stays fixed and comparable.
        perm = torch.from_numpy(np.random.permutation(n_train))
        epoch_loss = 0.0

        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            X_batch = X_t[idx]
            Y_batch = Y_t[idx]

            optimizer.zero_grad(set_to_none=True)
            # 0.5 * mean over (batch, out); matches nnzig's per-batch lossE.
            loss = 0.5 * criterion(model(X_batch), Y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Match nnzig: process the remainder batch when it exists. Its loss is
        # the mean over (remainder, out) and is counted once toward the epoch
        # average.
        if n_remainder != 0:
            idx = perm[n_batches * batch_size : n_train]
            X_batch = X_t[idx]
            Y_batch = Y_t[idx]

            optimizer.zero_grad(set_to_none=True)
            loss = 0.5 * criterion(model(X_batch), Y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # nnzig divides the accumulated batch losses by the number of full
        # batches (nBatchesF), so do the same here.
        train_losses[epoch] = epoch_loss / n_batches if n_batches > 0 else 0.0

        model.eval()
        with torch.no_grad():
            # 0.5 * mean over (val, out); matches nnzig's validation loss.
            val_losses[epoch] = (0.5 * criterion(model(X_v), Y_v)).item()

    write_losses_binary(
        config["outputs"]["losses_pytorch"], train_losses, val_losses, precision_bits
    )
    print("[info] Done!")
