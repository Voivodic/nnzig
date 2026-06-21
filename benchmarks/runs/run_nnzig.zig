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
    const data = try nn.loadData(fileName);
    defer allocator.free(data[0]);
    defer allocator.free(data[1]);

    // Normalize the dataset
    std.log.info("Computing normalization...", .{});
    try nn.computeNormalization(data[0], data[1]);
    try nn.normalize(data[0], data[1]);

    // Train the network
    std.log.info("Training network...", .{});
    const _bench_train_t0 = std.Io.Timestamp.now(ioContext, .awake);
    try nn.train(data[0], data[1]);
    const _bench_train_seconds: f64 =
        @as(f64, @floatFromInt(_bench_train_t0.durationTo(std.Io.Timestamp.now(ioContext, .awake)).nanoseconds)) /
        1_000_000_000.0;
    std.log.info("NNBENCH_TRAIN_SECONDS:{d:.6}", .{_bench_train_seconds});

    // Save the losses
    const lossesFileName = "benchmarks/losses/losses_zig.bin";
    try nn.saveLosses(lossesFileName);

    std.log.info("Done!", .{});
}
