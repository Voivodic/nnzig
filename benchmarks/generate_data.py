import numpy as np
import struct


def write_numpy_binary(filename, precision_bits, data_array):
    """
    Writes a header and a NumPy array of data points to a binary file.

    Header format:
    - 3 x u64 -> Precision, Count, Dimension
    Followed by:
    - Contiguous binary data from the NumPy array
    """
    # Ensure data is a NumPy array
    data_array = np.asarray(data_array)

    # Extract dimensions
    num_points = data_array.shape[0]
    dimension = data_array.shape[1] if data_array.ndim > 1 else 1

    # Match precision and cast array to appropriate type
    if precision_bits == 64:
        data_array = data_array.astype(np.float64)
    elif precision_bits == 32:
        data_array = data_array.astype(np.float32)
    else:
        raise ValueError("Precision bits must be 32 or 64.")

    # '<QQQ' means Little-Endian, 3x Unsigned 64-bit integers
    header_format = "<QQQ"
    header_bytes = struct.pack(header_format, precision_bits, num_points, dimension)

    with open(filename, "wb") as f:
        # 1. Write the 24-byte header
        f.write(header_bytes)
        # 2. Write all data points at once using NumPy's fast tobytes()
        f.write(data_array.tobytes())

    print(f"Successfully saved {num_points} points (Dim: {dimension}) to '{filename}'.")


def read_numpy_binary(filename):
    """Reads the binary file back into a structured NumPy array."""
    header_format = "<QQQ"
    header_size = struct.calcsize(header_format)

    with open(filename, "rb") as f:
        # 1. Read and unpack header
        header_bytes = f.read(header_size)
        precision_bits, num_points, dimension = struct.unpack(
            header_format, header_bytes
        )

        print("\n--- Header Info ---")
        print(f"Precision: {precision_bits} bits")
        print(f"Points:    {num_points}")
        print(f"Dimension: {dimension}\n")

        # 2. Determine NumPy dtype
        dtype = np.float64 if precision_bits == 64 else np.float32

        # 3. Read the rest of the file directly into a NumPy array
        # count=-1 reads all remaining data
        data = np.frombuffer(f.read(), dtype=dtype)

        # Reshape to restore original (points, dimension) structure
        return data.reshape((num_points, dimension))


# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy NumPy array of 5 points in 3D space
    # (Shape: 5 rows, 3 columns)
    np_data = np.array(
        [
            [1.1, 1.2, 1.3],
            [2.1, 2.2, 2.3],
            [3.1, 3.2, 3.3],
            [4.1, 4.2, 4.3],
            [5.1, 5.2, 5.3],
        ]
    )

    filename = "numpy_dataset.bin"

    # Save using 32-bit precision as an example
    write_numpy_binary(filename, precision_bits=32, data_array=np_data)

    # Read it back
    loaded_data = read_numpy_binary(filename)

    print("Original NumPy Array:")
    print(np_data)
    print("Loaded NumPy Array:")
    print(loaded_data)
