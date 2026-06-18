import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import os
import struct

from config import load_config

# Precision (in bits) -> (numpy dtype, jax dtype).
_PRECISION = {
    16: (np.float16, jnp.float16),
    32: (np.float32, jnp.float32),
    64: (np.float64, jnp.float64),
}

_ACTIVATIONS = {
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "tanh": jnp.tanh,
    "none": lambda x: x,
}

# reduction="mean" semantics matching torch.nn.MSELoss(reduction="mean"):
# the mean of the squared error over (batch, nOut). Combined with the
# 0.5 * ... factor applied at the call site, this matches nnzig's per-batch
# lossE (whose MSE kernel divides by nOut and whose backProp divides by
# batchSize).
_LOSS_FUNCS = {
    "MSE": lambda pred, y: jnp.mean((pred - y) ** 2),
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


class MLP(eqx.Module):
    """Multi-layer perceptron matching nnzig's mlp.zig layer stack.

    A Linear layer is followed by its activation in lock-step (the final
    activation, usually 'none', is applied to the last Linear as well),
    mirroring src/layers/mlp.zig which iterates activations alongside the
    weight matrices.
    """
    layers: list
    # Stored as static metadata so it is part of the PyTree treedef (not a
    # differentiable leaf) and is invisible to optax / eqx.filter_grad.
    activations: tuple = eqx.field(static=True)

    def __init__(self, layer_sizes, activations, *, key):
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                "activations must have len(layer_sizes) - 1 entries"
            )
        bad = [a for a in activations if a not in _ACTIVATIONS]
        if bad:
            raise ValueError("Unsupported activation(s): " + str(bad))
        keys = jrandom.split(key, len(layer_sizes) - 1)
        self.layers = [
            eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=keys[i])
            for i in range(len(layer_sizes) - 1)
        ]
        self.activations = tuple(activations)

    def __call__(self, x):
        for layer, act_name in zip(self.layers, self.activations):
            x = layer(x)
            x = _ACTIVATIONS[act_name](x)
        return x


def init_normal(model, key):
    """Replace every Linear weight and bias with an independent sample from
    N(0, 1), matching nnzig's weight init and run_pytorch's
    ``torch.nn.init.normal_(p, mean=0.0, std=1.0)``."""
    new_layers = []
    for layer in model.layers:
        key, w_key, b_key = jrandom.split(key, 3)
        new_weight = jrandom.normal(
            w_key, layer.weight.shape, dtype=layer.weight.dtype
        )
        # eqx.nn.Linear always has a bias here (use_bias=True default), so
        # layer.bias is an array, not None.
        new_bias = jrandom.normal(
            b_key, layer.bias.shape, dtype=layer.bias.dtype
        )
        new_layers.append(
            eqx.tree_at(
                lambda l: (l.weight, l.bias), layer, (new_weight, new_bias)
            )
        )
    return eqx.tree_at(lambda m: m.layers, model, new_layers)


if __name__ == "__main__":
    config = load_config()
    network = config["network"]
    training = config["training"]

    precision_bits = network["precision"]
    np_dtype, jax_dtype = _PRECISION[precision_bits]

    # JAX defaults to float32; enable x64 so float64 weights/inputs work
    # when precision=64. float16 arrays just use jnp.float16 directly.
    if precision_bits == 64:
        jax.config.update("jax_enable_x64", True)

    print("[info] Initializing NN...")

    seed = training["seed"]
    # Split the seed into one key for weight init and one for the per-epoch
    # shuffling RNG, mirroring run_pytorch's use of the same training.seed
    # for both torch.manual_seed and np.random.seed.
    init_key, shuffle_key = jrandom.split(jrandom.PRNGKey(seed))

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

    X_t = jnp.asarray(X[:n_train], dtype=jax_dtype)
    Y_t = jnp.asarray(Y[:n_train], dtype=jax_dtype)
    X_v = jnp.asarray(X[n_train:n_train + n_val], dtype=jax_dtype)
    Y_v = jnp.asarray(Y[n_train:n_train + n_val], dtype=jax_dtype)
    del X, Y

    print("[info] Training network...")

    model = MLP(network["nNeurons"], network["activations"], key=init_key)
    # model = init_normal(model, init_key)

    loss_name = network["lossFunc"]
    if loss_name not in _LOSS_FUNCS:
        raise ValueError("Unsupported loss function: " + loss_name)
    loss_fn = _LOSS_FUNCS[loss_name]

    # NOTE: optax.adam uses a global step counter (incremented every
    # optimizer.update() call). nnzig instead uses a per-epoch counter
    # t = epoch+1 for every batch within an epoch. This is the same
    # algorithmic difference as run_pytorch.py (whose torch.optim.Adam also
    # uses a global step counter); both use the same lr/beta1/beta2/eps.
    optimizer = optax.adam(
        learning_rate=training["lr"],
        b1=training["beta1"],
        b2=training["beta2"],
        eps=training["eps"],
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    n_epochs = training["nEpochs"]
    batch_size = training["batchSize"]
    n_batches = n_train // batch_size
    n_remainder = n_train % batch_size
    train_losses = np.empty(n_epochs, dtype=np_dtype)
    val_losses = np.empty(n_epochs, dtype=np_dtype)

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        # x: (batch, in), y: (batch, out). vmap the forward over the batch
        # axis. 0.5 * mean over (batch, out); matches nnzig's per-batch loss.
        pred = jax.vmap(model)(x)
        return 0.5 * loss_fn(pred, y)

    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss, grads = compute_loss(model, x, y)
        # opt_state was initialized against the filtered (inexact-array)
        # pytree, so filter grads the same way before handing them to optax.
        grads = eqx.filter(grads, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    @eqx.filter_jit
    def eval_loss(model, x, y):
        pred = jax.vmap(model)(x)
        # 0.5 * mean over (val, out); matches nnzig's validation loss.
        return 0.5 * loss_fn(pred, y)

    for epoch in range(n_epochs):
        # Match nnzig: shuffle only the training indices each epoch so the
        # validation set stays fixed and comparable.
        shuffle_key, epoch_key = jrandom.split(shuffle_key)
        perm = jrandom.permutation(epoch_key, n_train)
        epoch_loss = 0.0

        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            X_batch = X_t[idx]
            Y_batch = Y_t[idx]

            loss, model, opt_state = make_step(
                model, X_batch, Y_batch, opt_state
            )
            epoch_loss += float(loss)

        # Match nnzig: process the remainder batch when it exists. Its loss is
        # the mean over (remainder, out) and is counted once toward the epoch
        # average.
        if n_remainder != 0:
            idx = perm[n_batches * batch_size : n_train]
            X_batch = X_t[idx]
            Y_batch = Y_t[idx]

            loss, model, opt_state = make_step(
                model, X_batch, Y_batch, opt_state
            )
            epoch_loss += float(loss)

        # nnzig divides the accumulated batch losses by the number of full
        # batches (nBatchesF), so do the same here.
        train_losses[epoch] = epoch_loss / n_batches if n_batches > 0 else 0.0
        val_losses[epoch] = float(eval_loss(model, X_v, Y_v))

    write_losses_binary(
        config["outputs"]["losses_equinox"],
        train_losses,
        val_losses,
        precision_bits,
    )
    print("[info] Done!")
