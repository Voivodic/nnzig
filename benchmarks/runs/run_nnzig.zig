const std = @import("std");
const nnzig = @import("nnzig");
const params = @import("params");

pub fn main(init: std.process.Init) !void {
    // Get the io context from init
    const ioContext = init.io;

    // Get the allocator from init
    const allocator = init.gpa;

    // Initialize the NNN struct
    std.log.info("Initializing NN...", .{});
    var nn: nnzig.NN = try nnzig.NN.init(allocator, ioContext);
    defer nn.deinit();

    // Set the parameters of the dataset
    const fileName = "benchmarks/dataset_benchmark.bin";

    // Load the dataset
    std.log.info("Loading dataset...", .{});
    const results = try nn.loadData(fileName);
    const dataIn = results[0];
    defer allocator.free(dataIn);
    const dataOut = results[1];
    defer allocator.free(dataOut);

    // Normalize the dataset
    std.log.info("Computing normalization...", .{});
    try nn.computeNormalization(dataIn, dataOut);
    try nn.normalize(dataIn, dataOut);

    // Train the network
    std.log.info("Training network...", .{});
    const _bench_train_t0 = std.Io.Timestamp.now(ioContext, .awake);
    try nn.train(dataIn, dataOut);
    // Training-only wall time, printed for bench_resources.py to parse. This
    // excludes init + dataset load + normalization + loss saving, removing
    // the fixed startup cost from the reported time (matches the Python
    // runners, which time only their training loop). Zig 0.16 moved monotonic
    // timing into the I/O layer (std.Io.Clock), so read the .awake clock
    // (CLOCK_MONOTONIC on Linux) through the ioContext the harness passes in.
    const _bench_train_seconds: f64 =
        @as(f64, @floatFromInt(_bench_train_t0.durationTo(std.Io.Timestamp.now(ioContext, .awake)).nanoseconds)) /
        1_000_000_000.0;
    std.log.info("NNBENCH_TRAIN_SECONDS:{d:.6}", .{_bench_train_seconds});

    // Save the losses
    const lossesFileName = "benchmarks/losses/losses_zig.bin";
    try nn.saveLosses(lossesFileName);

    std.log.info("Done!", .{});
}
