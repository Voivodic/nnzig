//! This module reads and parses the parameters from ../../params.json at comptime
const std = @import("std");
const config = @import("config");

/// Enum of the possible activation functions
pub const Activation = enum(u8) {
    none,
    relu,
    tanh,
    sigmoid,
};

/// Enum of the possible loss functions
pub const Loss = enum(u8) {
    MSE,
};

/// Enum of the possible ways to normalize the data
pub const NormType = enum(u8) {
    meanStd,
};

/// The structure must match the JSON keys exactly
const Config = struct {
    numThreads: usize,
    nNeurons: []const usize,
    activations: []const Activation,
    lossFunc: Loss,
    normType: NormType,
    seed: u64,
    rTrain: f32,
    rVal: f32,
    eps: f32,
    beta1: f32,
    beta2: f32,
    lr: f32,
    nEpochs: usize,
    batchSize: usize,
    printEvery: usize,
};

// Parse the JSON at comptime
pub const params = blk: {
    const raw_data = @embedFile(config.params_path);
    
    // Uses parseFromSliceLeaky because we are at comptime;
    const parsed = std.json.parseFromSliceLeaky(
        Config,
        std.heap.page_allocator,
        raw_data,
        .{ .ignore_unknown_fields = true },
    ) catch |err| {
        @compileError("JSON Parse Error: " ++ @errorName(err));
    };

    break :blk parsed;
};

// Logic for derived parameters (like rTest) remains in Zig code
pub const rTest: f32 = 1.0 - params.rTrain - params.rVal;

