//! This module defines the activation functions and their derivatives

// Import the modules used
const std = @import("std");
const params = @import("params").params;
const errors = @import("errors");
const err = errors.activationError;

// The trivial activation function
fn none(comptime T: type, df: []T) void {
    for (df) |*d| {
        d.* = 1.0;
    }
}

// The relu activation funciton
fn relu(comptime T: type, slice: []T, df: []T) void {
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
fn tanh(comptime T: type, slice: []T, df: []T) void {
    for (slice, df) |*elem, *d| {
        const exp = @exp(elem.*);
        const invexp = 1.0 / exp;

        elem.* = (exp * exp - 1.0) / (exp * exp + 1.0);
        d.* = 4.0 / (exp + invexp) * (exp + invexp);
    }
}

// The sigmoid activation function
fn sigmoid(comptime T: type, slice: []T, df: []T) void {
    for (slice, df) |*elem, *d| {
        const sigma = 1.0 / (1.0 + @exp(-elem.*));

        elem.* = sigma;
        d.* = sigma * (1.0 - sigma);
    }
}

/// Apply the activation function to each element of the slice and compute its gradient
pub fn activateElements(comptime T: type, input: []T, df: []T, act: params.activation) !void {
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
                if (std.Thread.spawn(.{}, none, .{ T, df[start..end] })) |t| {
                    thread.* = t;
                } else |_| {
                    std.debug.print("Error in spawning thread {}!\n", .{i});
                    return err.threadRun;
                }
            },
            // ReLU
            params.activation.relu => {
                if (std.Thread.spawn(.{}, relu, .{ T, input[start..end], df[start..end] })) |t| {
                    thread.* = t;
                } else |_| {
                    std.debug.print("Error in spawning thread {}!\n", .{i});
                    return err.threadRun;
                }
            },
            // Tanh
            params.activation.tanh => {
                if (std.Thread.spawn(.{}, tanh, .{ T, input[start..end], df[start..end] })) |t| {
                    thread.* = t;
                } else |_| {
                    std.debug.print("Error in spawning thread {}!\n", .{i});
                    return err.threadRun;
                }
            },
            // Sigmoid
            params.activation.sigmoid => {
                if (std.Thread.spawn(.{}, sigmoid, .{ T, input[start..end], df[start..end] })) |t| {
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
