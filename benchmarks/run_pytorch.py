import torch
import numpy as np
import struct

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



