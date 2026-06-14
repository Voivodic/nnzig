# NNzig

Compile-time-configured neural network library in Zig with C++/Eigen linear-algebra kernels.

## Toolchain

- **Zig 0.16.0** exactly (CI pins it; installed locally). Not 0.15 — the I/O APIs below are 0.16-specific.
- **Eigen 5.0.0** is a fetched build dependency (`build.zig.zon`); the first build downloads it. The Eigen module links libc + libcpp.

## Commands

- `zig build test --summary all` — the only verification step. Zig has no separate lint/typecheck; run this after every change.
- `zig build docs` — emits `zig-out/docs` (deployed to GitHub Pages from `main`).
- `zig build benchmark` — builds **and runs** the benchmark. Requires `benchmarks/dataset_benchmark.bin` (generate first, see below). Plain `zig build` also installs the benchmark executable.
- CI (`.github/workflows/`) runs `test` on every PR/push to `main`; `docs` builds on every push and deploys on merge to `main`.

There is no formatter/linter step.

## Configuration model — read this before editing params

Network shape, precision, and training hyperparameters are **compile-time constants** parsed from a ZON file. There are **three independent copies**, one per build context — editing one does not change the others:

| Context         | Params file            |
| --------------- | ---------------------- |
| docs / library  | `params.zon`           |
| tests           | `tests/params.zon`     |
| benchmark       | `benchmarks/params.zon`|

`build.zig` builds a separate module tree for each. `src/core/params.zig` parses the ZON and exposes typed `pub const` values. Key facts:

- `.precision` (16/32/64) selects the float type **and** is passed to the C++ kernels as `-DFLOAT_PRECISION=<n>`. This is handled by `build.zig` (`std.fmt.comptimePrint`); do not try to make precision a runtime value.
- The float type is the alias **`T`** in the `params` module (`params.T`). It is used repo-wide — do not introduce a per-file float type.
- Missing/invalid fields raise `@compileError` in `params.zig`, so config mistakes surface at build time.

## Architecture

- `src/nnzig.zig` — public API: the `NN` struct (`init`/`deinit`, `forward`, `train`, normalization, `save/loadWeights`, `saveLosses`). Training is mini-batch gradient descent with the Adam optimizer.
- `src/eigen/` — **the heavy linear algebra lives here, in C++/Eigen** (`linalg.cpp`, `activations.cpp`) wrapped by `wrappers.zig`. Forward/backward/gradient kernels call through these extern functions; do not reimplement them in pure Zig.
- `src/core/` — `params.zig` (config), `errors.zig` (shared error enum), `normalizations.zig`.
- `src/cpu/` — pure-Zig activations and losses.
- `src/layers/mlp.zig` — MLP layer.
- `src/io/binary.zig` — binary weight/loss/dataset I/O.

Module **import names differ from filenames** (defined in the `Tree` in `build.zig`): `act`, `loss`, `eigen`, `mlp`, `params`, `norms`, `io`, `errors`, `nnzig`, `paramsFile`. Reference these names, not the file paths.

## Zig 0.16 conventions used here

This codebase targets the 0.16 I/O model, not the legacy one:

- Entry points: `pub fn main(init: std.process.Init) !void` — take `init.io` (the `std.Io` context) and `init.gpa` (the allocator). See `benchmarks/run_nnzig.zig`.
- Tests obtain the I/O context from `std.testing.io`, not a constructed `std.io`.
- File operations go through `std.Io.Dir` (e.g., `std.Io.Dir.cwd()`, `cwd.deleteFile(ioContext, path)`).

Tests are written **inline** in each source file as `test "..."` blocks and gathered via `comptime { std.testing.refAllDecls(@This()); }`. `build.zig` runs a separate test executable per module (params, eigen, io, norms, activation, loss, mlp, nnzig) plus an integration suite in `tests/tests.zig`.

## Benchmark setup

`zig build benchmark` reads `benchmarks/dataset_benchmark.bin` (gitignored). Generate it first:

- `nix run ./benchmarks#generate-data` (the flake also provides `run-pytorch` and `plot-losses` for the PyTorch comparison).

## graphify

This project has a knowledge graph at `graphify-out/` with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than `GRAPH_REPORT.md` or raw grep output.
- Dirty `graphify-out/` files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
