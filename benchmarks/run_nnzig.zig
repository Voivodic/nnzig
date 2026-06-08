const std = @import("std");
const nnzig = @import("nnzig");
const io = @import("io");
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
    const nData: usize = 100;
    const dimIn: usize = 2;
    const dimOut: usize = 2;

    // Load the dataset
    std.log.info("Loading dataset...", .{});
    var dataIn: [nData * dimIn]f32 = undefined;
    var dataOut: [nData * dimOut]f32 = undefined;
    try io.loadData(ioContext, fileName, &dataIn, &dataOut);

    // Normalize the dataset
    std.log.info("Computing normalization...", .{});
    try nn.computeNormalization(&dataIn, &dataOut);
    try nn.normalize(&dataIn, &dataOut);

    // Train the network
    std.log.info("Training network...", .{});
    try nn.train(&dataIn, &dataOut);

    // Save the losses
    const lossesFileName = "benchmarks/losses_zig.bin";
    try nn.saveLosses(lossesFileName);

    std.log.info("Done!", .{});
}
