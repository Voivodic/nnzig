import numpy as np
import os
import struct
import tensorflow as tf

from config import load_config

# Silence TensorFlow's verbose C++/abseil logging so the benchmark output
# matches the clean "[info] ..." lines of run_pytorch / run_equinox.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Keras 3 (shipped as a separate package in TF >= 2.16) can otherwise default
# to a non-TF backend (jax/torch). Force tensorflow so the Dense layers and
# GradientTape run on the TF engine, matching PyTorch / Equinox / nnzig.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# Precision (in bits) -> (numpy dtype, tensorflow dtype).
_PRECISION = {
    16: (np.float16, tf.float16),
    32: (np.float32, tf.float32),
    64: (np.float64, tf.float64),
}

# Mean of the squared error over ALL elements (batch * nOut), identical to
# torch.nn.MSELoss(reduction="mean"). Combined with the 0.5 * ... factor
# applied at the call site, this matches nnzig's per-batch lossE (whose MSE
# kernel divides by nOut and whose backProp divides by batchSize).
_LOSS_FUNCS = {
    "MSE": lambda pred, y: tf.reduce_mean(tf.square(pred - y)),
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
    header_bytes = struct.pack(header_format, precision_bytes, len(train_losses), 1, 1)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "wb") as f:
        f.write(header_bytes)
        f.write(train_losses.astype(np_dtype).tobytes())
        f.write(val_losses.astype(np_dtype).tobytes())


class MLP(tf.keras.Sequential):
    """Multi-layer perceptron matching nnzig's mlp.zig layer stack.

    A Dense layer is followed by its activation in lock-step (the final
    activation, usually 'none', is applied to the last Dense as well),
    mirroring src/layers/mlp.zig which iterates activations alongside the
    weight matrices. This also matches run_pytorch's
    ``[Linear, ReLU, Linear, ReLU, ...]`` Sequential.
    """

    def __init__(self, layer_sizes, activations, tf_dtype):
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("activations must have len(layer_sizes) - 1 entries")
        layers = []
        for i in range(len(layer_sizes) - 1):
            # Keras 3's Sequential.__init__ does not accept a `dtype` kwarg
            # (unlike Keras 2), so set the compute/variable dtype on each
            # Dense layer directly. This mirrors run_pytorch's
            # torch.set_default_dtype + .to(torch_dtype) so nothing silently
            # falls back to f64. Activation layers are dtype-transparent
            # (they preserve their input's dtype).
            layers.append(tf.keras.layers.Dense(layer_sizes[i + 1], dtype=tf_dtype))
            act = activations[i]
            if act == "relu":
                layers.append(tf.keras.layers.ReLU())
            elif act == "sigmoid":
                layers.append(tf.keras.layers.Activation("sigmoid"))
            elif act == "tanh":
                layers.append(tf.keras.layers.Activation("tanh"))
            elif act == "none":
                pass
            else:
                raise ValueError("Unsupported activation: " + act)
        super().__init__(layers)
        # Build eagerly against the known input shape (batch, nNeurons[0]) so
        # model.trainable_variables is populated before training. Uses an
        # explicit build() instead of the legacy Dense(input_shape=...) kwarg
        # (dropped from the Keras 3 Dense signature). The model takes input of
        # shape (batch, nNeurons[0]) and outputs (batch, nNeurons[-1]).
        self.build((None, layer_sizes[0]))


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")

    config = load_config()
    network = config["network"]
    training = config["training"]

    precision_bits = network["precision"]
    np_dtype, tf_dtype = _PRECISION[precision_bits]

    print("[info] Initializing NN...")

    seed = training["seed"]
    # Mirror run_pytorch, which sets both the framework RNG (torch.manual_seed)
    # and numpy's (np.random.seed). set_random_seed seeds TF/keras; we still
    # seed numpy explicitly because the per-epoch shuffle uses np.random.
    tf.keras.utils.set_random_seed(seed)
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

    X_t = tf.constant(X[:n_train], dtype=tf_dtype)
    Y_t = tf.constant(Y[:n_train], dtype=tf_dtype)
    X_v = tf.constant(X[n_train : n_train + n_val], dtype=tf_dtype)
    Y_v = tf.constant(Y[n_train : n_train + n_val], dtype=tf_dtype)
    del X, Y

    print("[info] Training network...")

    model = MLP(network["nNeurons"], network["activations"], tf_dtype)
    # Match nnzig's init: every weight and bias drawn from N(0, 1).
    # for layer in model.layers:
    #     if isinstance(layer, tf.keras.layers.Dense):
    #         w = layer.kernel.numpy()
    #         b = layer.bias.numpy()
    #         layer.kernel.assign(tf.random.normal(w.shape, mean=0.0, stddev=1.0, dtype=tf_dtype))
    #         layer.bias.assign(tf.random.normal(b.shape, mean=0.0, stddev=1.0, dtype=tf_dtype))

    loss_name = network["lossFunc"]
    if loss_name not in _LOSS_FUNCS:
        raise ValueError("Unsupported loss function: " + loss_name)
    loss_fn = _LOSS_FUNCS[loss_name]

    # NOTE: tf.keras.optimizers.Adam uses a global step counter (incremented
    # every optimizer.apply_gradients() call). nnzig instead uses a per-epoch
    # counter t = epoch+1 for every batch within an epoch. This is the same
    # algorithmic difference as run_pytorch.py (whose torch.optim.Adam also
    # uses a global step counter); both use the same lr/beta1/beta2/eps.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=training["lr"],
        beta_1=training["beta1"],
        beta_2=training["beta2"],
        epsilon=training["eps"],
    )

    n_epochs = training["nEpochs"]
    batch_size = training["batchSize"]
    n_batches = n_train // batch_size
    n_remainder = n_train % batch_size
    train_losses = np.empty(n_epochs, dtype=np_dtype)
    val_losses = np.empty(n_epochs, dtype=np_dtype)

    for epoch in range(n_epochs):
        # Match nnzig: shuffle only the training indices each epoch so the
        # validation set stays fixed and comparable.
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0

        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            X_batch = tf.gather(X_t, idx)
            Y_batch = tf.gather(Y_t, idx)

            with tf.GradientTape() as tape:
                # 0.5 * mean over (batch, out); matches nnzig's per-batch lossE.
                loss = 0.5 * loss_fn(model(X_batch), Y_batch)
            grads = tape.gradient(loss, model.trainable_variables)
            # grads and trainable_variables are equal-length by construction
            # (tape.gradient yields one tensor per variable).
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables, strict=False)
            )

            epoch_loss += float(loss)

        # Match nnzig: process the remainder batch when it exists. Its loss is
        # the mean over (remainder, out) and is counted once toward the epoch
        # average.
        if n_remainder != 0:
            idx = perm[n_batches * batch_size : n_train]
            X_batch = tf.gather(X_t, idx)
            Y_batch = tf.gather(Y_t, idx)

            with tf.GradientTape() as tape:
                loss = 0.5 * loss_fn(model(X_batch), Y_batch)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables, strict=False)
            )

            epoch_loss += float(loss)

        # nnzig divides the accumulated batch losses by the number of full
        # batches (nBatchesF), so do the same here.
        train_losses[epoch] = epoch_loss / n_batches if n_batches > 0 else 0.0

        # No GradientTape here: pure forward pass for the validation loss.
        # 0.5 * mean over (val, out); matches nnzig's validation loss.
        val_losses[epoch] = float(0.5 * loss_fn(model(X_v), Y_v))

    write_losses_binary(
        config["outputs"]["losses_tensorflow"], train_losses, val_losses, precision_bits
    )
    print("[info] Done!")
