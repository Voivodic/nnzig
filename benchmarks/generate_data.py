import numpy as np
import struct

from config import load_config, write_params_zon

# Precision (in bits) -> number of bytes and numpy dtype.
_PRECISION = {
    16: (2, np.float16),
    32: (4, np.float32),
    64: (8, np.float64),
}


def generate_dataset(config):
    dataset = config["dataset"]
    network = config["network"]

    num_points = dataset["num_points"]
    dim_in = network["nNeurons"][0]
    dim_out = network["nNeurons"][-1]
    precision_bits = network["precision"]

    if precision_bits not in _PRECISION:
        raise ValueError("Invalid precision: " + str(precision_bits))
    precision_bytes, np_dtype = _PRECISION[precision_bits]

    # Create the numpy arrays
    x_data = np.random.rand(num_points, dim_in)
    y_data = np.zeros((num_points, dim_out))
    coef = np.random.rand(dim_in)
    for dim_y in range(dim_out):
        y_data[:, dim_y] = ((dim_y + 1.0) / dim_out) * np.ones(num_points)
        for dim_x in range(dim_in):
            y_data[:, dim_y] += coef[dim_x] * x_data[:, dim_x] ** ((dim_x + 1) / dim_in)
    y_data /= dim_in + 1

    # Convert the numpy arrays to the desired precision
    x_data = x_data.astype(np_dtype)
    y_data = y_data.astype(np_dtype)

    # '<QQQQ' means Little-Endian, 4x Unsigned 64-bit integers
    header_format = "<QQQQ"
    header_bytes = struct.pack(
        header_format, precision_bytes, num_points, dim_in, dim_out
    )

    # Save the dataset
    file_name = dataset["file"]
    with open(file_name, "wb") as f:
        # 1. Write the 32-byte header
        f.write(header_bytes)

        # 2. Write all data points at once using NumPy's fast tobytes()
        f.write(x_data.tobytes())
        f.write(y_data.tobytes())

    print("Successfully saved " + str(num_points) + " points to '" + file_name + "'.")


if __name__ == "__main__":
    config = load_config()

    # Make the dataset reproducible from the configured dataset seed (kept
    # independent of training.seed so the dataset stays fixed when sweeping
    # optimizer seeds).
    np.random.seed(config["dataset"]["seed"])

    generate_dataset(config)
    write_params_zon(config)
