# NNzig

A small, compile-time-configured neural network library written in Zig, with the
heavy linear algebra offloaded to C++/Eigen kernels. Network shape, numeric
precision, and training hyperparameters are all fixed at compile time from a
single ZON config file.

## Features

- Multi-layer perceptron (MLP) with configurable depth and width.
- Mini-batch gradient descent with the Adam optimizer.
- Plugggable activation functions (ReLU, tanh, sigmoid) and losses (MSE).
- Z-score input/output normalization.
- Binary save/load of weights and per-epoch loss curves.
- Compile-time selectable float precision (16/32/64-bit).
- C++/Eigen-accelerated forward, backward, and gradient kernels.

## Requirements

- **[Zig](https://ziglang.org) 0.16.0** — the exact version used in CI.
- A C++ toolchain with libc++ (for the Eigen kernels).
- **[Eigen](https://eigen.tuxfamily.org) 5.0.0** is fetched automatically on the
  first build via `build.zig.zon`; no manual installation needed.

## Usage

1. Configure the network in `params.zon` (network shape, precision, training
   hyperparameters, etc.).
2. Import the library and drive it through the `NN` struct:

```zig
const std = @import("std");
const nnzig = @import("nnzig");
const io = @import("io");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // Create the network (shape/precision come from params.zon)
    var nn = try nnzig.NN.init(allocator, init.io);
    defer nn.deinit();

    // Load training data, then normalize it
    var inputs: [200]f32 = undefined;
    var outputs: [200]f32 = undefined;
    try io.loadData(init.io, "data.bin", &inputs, &outputs);
    try nn.computeNormalization(&inputs, &outputs);
    try nn.normalize(&inputs, &outputs);

    // Train and save the loss history
    try nn.train(&inputs, &outputs);
    try nn.saveLosses("losses.bin");

    // Run a forward pass
    const x = [_]f32{ 1.0, 2.0 };
    const y = try nn.forward(&x);
}
```

The public API lives in `src/nnzig.zig`: `init`/`deinit`, `forward`, `train`,
`computeNormalization`/`normalize`/`denormalize`, `save/loadWeights`, and
`saveLosses`. A complete end-to-end example is in `benchmarks/runs/run_nnzig.zig`.

## Configuration

All network behavior is set at compile time in `params.zon`:

| Field           | Description                                              |
| --------------- | -------------------------------------------------------- |
| `precision`     | Float precision: `16`, `32`, or `64`                     |
| `nNeurons`      | Neurons per layer, e.g. `.{ 2, 4, 4, 2 }`                |
| `activations`   | Per-layer activations (`relu`, `tanh`, `sigmoid`, `none`)|
| `lossFunc`      | Loss function (`MSE`)                                    |
| `normalization` | Normalization method (`meanStd`)                         |
| `nEpochs`       | Number of training epochs                                |
| `batchSize`     | Mini-batch size                                          |
| `lr`, `beta1`, `beta2`, `eps` | Adam optimizer hyperparameters            |
| `rTrain`, `rVal`| Train/validation split fractions                         |

Invalid values are caught at build time. See `params.zon` for the full set.

## Testing

```sh
nix run .#test -- --summary all
OR
zig build test --summary all
```

Tests are written inline in each source file and cover every module.

## Documentation

```sh
nix run .#docs
OR
zig build docs
```

Generates API documentation in `zig-out/docs`, also published to GitHub Pages
from the `main` branch.

## Benchmarks

A benchmark compares NNzig against an equivalent PyTorch model.

```sh
nix run .#benchmark
OR
zig build benchmark
```

It reads `benchmarks/dataset_benchmark.bin`, which is gitignored. Generate it
first:

```sh
nix run .#generate-data
```

The benchmark suite includes a PyTorch, Equinox, and TensorFlow references
and loss-plotting script. Use [Nix](https://nixos.org), the provided flake
sets up the Python environment:

```sh
nix run .#generate-data   # generate the dataset
nix run .#run-pytorch     # run the PyTorch reference
nix run .#run-equinox     # run the Equinox reference
nic run .#run-tensorflow      # run the TensorFlow reference
nix run .#plot-losses     # plot both loss curves
nix run .#bench-resources # sweep N: time & peak-RSS vs PyTorch/Equinox/TF
nix run .#plot-resources  # plot the resource-benchmark results
```

`bench-resources` builds nnzig through the same flake (via `.#default`) as the
Python libraries, so the whole sweep is reproducible under Nix with no local
Zig toolchain required.

## License

[MIT](LICENSE) © Rodrigo Voivodic
