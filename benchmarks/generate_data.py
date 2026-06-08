import numpy as np
import struct

# Create the dataset
if __name__ == "__main__":
    # Create a dummy NumPy array of 100 points in 2D space
    precision_bytes = 4
    N = 100
    dim = 2
    x_data = np.random.rand(N, dim)
    y_data = np.zeros((N, dim))
    y_data[:, 0] = (
        2.0
        + 1.2 * x_data[:, 0]
        - np.power(x_data[:, 1], 2)
        + np.exp(-3.0 * x_data[:, 0] - 2.0 * x_data[:, 1])
    )
    y_data[:, 1] = (
        1.4
        + 3.0 * x_data[:, 1]
        - np.power(x_data[:, 1], 2)
        + np.exp(-2.0 * x_data[:, 0] - 3.0 * x_data[:, 1])
    )
    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.float32)

    # Define the filename
    filename = "dataset_benchmark.bin"

    # '<QQQ' means Little-Endian, 3x Unsigned 64-bit integers
    header_format = "<QQQQ"
    header_bytes = struct.pack(header_format, precision_bytes, N, dim, dim)

    # Save the dataset
    with open(filename, "wb") as f:
        # 1. Write the 24-byte header
        f.write(header_bytes)

        # 2. Write all data points at once using NumPy's fast tobytes()
        f.write(x_data.tobytes())
        f.write(y_data.tobytes())

    print(f"Successfully saved {N} points to '{filename}'.")
