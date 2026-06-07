//! This module defines the activation functions and their derivatives

// Import the modules used
const std = @import("std");
const params = @import("params");
const errors = @import("errors");
const err = errors.activationError;
const fType = params.floatType;

// The trivial activation function
fn none(df: []fType) void {
    for (df) |*d| {
        d.* = 1.0;
    }
}

// The relu activation funciton
fn relu(slice: []fType, df: []fType) void {
    for (slice, df) |*elem, *d| {
        if (elem.* < 0.0) {
            elem.* = 0.0;
            d.* = 0.0;
        } else {
            d.* = 1.0;
        }
    }
}

// The tanh activation function
fn tanh(slice: []fType, df: []fType) void {
    for (slice, df) |*elem, *d| {
        const exp = @exp(elem.*);
        const invexp = 1.0 / exp;

        elem.* = (exp * exp - 1.0) / (exp * exp + 1.0);
        d.* = 4.0 / (exp + invexp) * (exp + invexp);
    }
}

// The sigmoid activation function
fn sigmoid(slice: []fType, df: []fType) void {
    for (slice, df) |*elem, *d| {
        const sigma = 1.0 / (1.0 + @exp(-elem.*));

        elem.* = sigma;
        d.* = sigma * (1.0 - sigma);
    }
}

/// Apply the activation function to each element of the slice and compute its gradient
pub fn activateElements(input: []fType, df: []fType, act: params.activation) !void {
    // Compute the chunck size
    const chunkSize: usize = (input.len + params.numThreads - 1) / params.numThreads;

    // Define the array of threads
    var threads: [params.numThreads]std.Thread = undefined;

    // Compute the chuncks for each thread
    for (&threads, 0..params.numThreads) |*thread, i| {
        const start: usize = i * chunkSize;
        const end: usize = @min(start + chunkSize, input.len);

        // Check the activation function to be used
        switch (act) {
            // Trivial
            params.activation.none => {
                if (std.Thread.spawn(.{}, none, .{df[start..end] })) |t| {
                    thread.* = t;
                } else |_| {
                    std.debug.print("Error in spawning thread {}!\n", .{i});
                    return err.threadRun;
                }
            },
            // ReLU
            params.activation.relu => {
                if (std.Thread.spawn(.{}, relu, .{input[start..end], df[start..end] })) |t| {
                    thread.* = t;
                } else |_| {
                    std.debug.print("Error in spawning thread {}!\n", .{i});
                    return err.threadRun;
                }
            },
            // Tanh
            params.activation.tanh => {
                if (std.Thread.spawn(.{}, tanh, .{input[start..end], df[start..end] })) |t| {
                    thread.* = t;
                } else |_| {
                    std.debug.print("Error in spawning thread {}!\n", .{i});
                    return err.threadRun;
                }
            },
            // Sigmoid
            params.activation.sigmoid => {
                if (std.Thread.spawn(.{}, sigmoid, .{input[start..end], df[start..end] })) |t| {
                    thread.* = t;
                } else |_| {
                    std.debug.print("Error in spawning thread {}!\n", .{i});
                    return err.threadRun;
                }
            },
        }
    }

    // Await all threads to finish
    for (&threads) |*thread| {
        thread.*.join();
    }
}

test "[act] activateElements none" {
    var input = [_]fType{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var df: [5]fType = undefined;

    try activateElements(&input, &df, params.activation.none);

    const testing = std.testing;
    for (input) |val| try testing.expect(std.math.isFinite(val));
    for (df) |val| try testing.expectEqual(@as(fType, 1.0), val);
}

test "[act] activateElements relu" {
    var input = [_]fType{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var df: [5]fType = undefined;

    try activateElements(&input, &df, params.activation.relu);

    const testing = std.testing;
    try testing.expectEqual(@as(fType, 0.0), input[0]);
    try testing.expectEqual(@as(fType, 0.0), input[1]);
    try testing.expectEqual(@as(fType, 0.0), input[2]);
    try testing.expectEqual(@as(fType, 1.0), input[3]);
    try testing.expectEqual(@as(fType, 2.0), input[4]);

    try testing.expectEqual(@as(fType, 0.0), df[0]);
    try testing.expectEqual(@as(fType, 0.0), df[1]);
    try testing.expectEqual(@as(fType, 1.0), df[2]);
    try testing.expectEqual(@as(fType, 1.0), df[3]);
    try testing.expectEqual(@as(fType, 1.0), df[4]);
}

test "[act] activateElements tanh" {
    var input = [_]fType{ 0.0 };
    var df: [1]fType = undefined;

    try activateElements(&input, &df, params.activation.tanh);

    const testing = std.testing;
    try testing.expectApproxEqAbs(@as(fType, 0.0), input[0], 1e-4);
    try testing.expect(df[0] > 0);
}

test "[act] activateElements sigmoid" {
    var input = [_]fType{ 0.0 };
    var df: [1]fType = undefined;

    try activateElements(&input, &df, params.activation.sigmoid);

    const testing = std.testing;
    try testing.expectApproxEqAbs(@as(fType, 0.5), input[0], 1e-4);
    try testing.expectApproxEqAbs(@as(fType, 0.25), df[0], 1e-4);
}
