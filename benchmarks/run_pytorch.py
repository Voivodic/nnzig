import torch
import numpy as np
import struct


def read_numpy_binary(filename):
    header_format = "<QQQQ"
    header_size = struct.calcsize(header_format)

    with open(filename, "rb") as f:
        header_bytes = f.read(header_size)
        precision_bits, num_points, dimIn, dimOut = struct.unpack(
            header_format, header_bytes
        )



        dtype = np.float64 if precision_bits == 64 else np.float32
        data = np.frombuffer(f.read(), dtype=dtype)

        input_size = num_points * dimIn
        input_data = data[:input_size].reshape((num_points, dimIn)).copy()
        output_data = data[input_size:].reshape((num_points, dimOut)).copy()

        return input_data, output_data, precision_bits


def write_losses_binary(filename, train_losses, val_losses, precision_bits=32):
    header_format = "<QQQQ"
    header_bytes = struct.pack(header_format, precision_bits, len(train_losses), 1, 1)
    with open(filename, "wb") as f:
        f.write(header_bytes)
        f.write(train_losses.tobytes())
        f.write(val_losses.tobytes())


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
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    print("[info] Initializing NN...")
    # nThreads = 1
    # torch.set_num_threads(nThreads)
    # torch.set_num_interop_threads(nThreads)

    seed = 12345
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("[info] Loading dataset...")
    X, Y, precision = read_numpy_binary("dataset_benchmark.bin")

    print("[info] Computing normalization...")
    X -= X.mean(axis=0)
    X /= X.std(axis=0) + 1e-8
    Y -= Y.mean(axis=0)
    Y /= Y.std(axis=0) + 1e-8

    N = X.shape[0]
    n_train = int(N * 0.7)
    n_val = int(N * 0.2)

    X_t = torch.tensor(X[:n_train], dtype=torch.float32)
    Y_t = torch.tensor(Y[:n_train], dtype=torch.float32)
    X_v = torch.tensor(X[n_train:n_train + n_val], dtype=torch.float32)
    Y_v = torch.tensor(Y[n_train:n_train + n_val], dtype=torch.float32)
    del X, Y

    print("[info] Training network...")

    model = MLP([2, 4, 4, 2], ["relu", "relu", "none"])
    for p in model.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=1.0)

    dimOut = 2
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8
    )

    n_epochs = 500
    batch_size = 50
    n_train_samples = X_t.shape[0]
    n_batches = n_train_samples // batch_size
    train_losses = np.empty(n_epochs, dtype=np.float32)
    val_losses = np.empty(n_epochs, dtype=np.float32)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end = start + batch_size
            X_batch = X_t[start:end]
            Y_batch = Y_t[start:end]

            optimizer.zero_grad(set_to_none=True)
            loss = 0.5 * criterion(model(X_batch), Y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() / (batch_size * dimOut)

        train_losses[epoch] = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_losses[epoch] = (0.5 * criterion(model(X_v), Y_v)).item() / (X_v.shape[0] * dimOut)

    write_losses_binary("losses_pytorch.bin", train_losses, val_losses, precision)
    print("[info] Done!")
