//! This module define the loss functions and their derivatives

// Imports the modules used
const std = @import("std");
const params = @import("params");
const errors = @import("errors");
const err = errors.lossError;
const T = params.T;

/// Computes the mean squared error loss and its derivatives: L = 0.5 * mean((pred - out)^2)
fn MSE(pred: []const T, out: []const T, dL: []T, value: *T) void {
    var loss: T = 0.0;

    // Compute the loss and its derivative
    for (pred, out, dL) |*p, *o, *dl| {
        dl.* = (p.* - o.*);
        loss += 0.5 * (dl.*) * (dl.*);
        dl.* = dl.* / @as(T, @floatFromInt(out.len));
    }

    value.* = loss / @as(T, @floatFromInt(out.len));
}

/// Function that computes the loss and its derivatives for a given choice of loss function
pub fn computeLoss(pred: []const T, out: []const T, dL: []T, lossFunc: params.loss) !T {
    // Compute the chunck size
    const chunkSize: usize = (pred.len + params.numThreads - 1) / params.numThreads;

    // Define the array of threads
    var threads: [params.numThreads]std.Thread = undefined;
    var lossThread: [params.numThreads]T = undefined;
    for (&lossThread) |*lt| {
        lt.* = 0.0;
    }

    // Compute the chuncks for each thread
    for (&threads, &lossThread, 0..params.numThreads) |*thread, *lt, i| {
        const start: usize = i * chunkSize;
        const end: usize = @min(start + chunkSize, pred.len);

        // Check which loss function to use
        switch (lossFunc) {
            params.loss.MSE => {
                if (std.Thread.spawn(.{}, MSE, .{ pred[start..end], out[start..end], dL[start..end], lt })) |t| {
                    thread.* = t;
                } else |_| {
                    std.log.err("Error in spawning thread {}!\n", .{i});
                    return err.threadRun;
                }
            },
        }
    }

    // Await all threads to finish
    for (&threads) |*thread| {
        thread.*.join();
    }

    // Compute the total final loss
    var totalLoss: T = 0.0;
    for (&lossThread) |*lt| {
        totalLoss += lt.*;
    }

    return totalLoss;
}

test "[loss] computeLoss MSE zero error" {
    const testing = std.testing;
    const nOut: usize = params.nNeurons[params.nNeurons.len - 1];

    var pred: [nOut]T = undefined;
    var out: [nOut]T = undefined;
    var dL: [nOut]T = undefined;

    for (&pred, &out) |*p, *o| {
        p.* = 1.0;
        o.* = 1.0;
    }

    const lossVal = try computeLoss(&pred, &out, &dL, params.loss.MSE);

    try testing.expectApproxEqAbs(@as(T, 0.0), lossVal, 1e-6);
    for (dL) |val| try testing.expectApproxEqAbs(@as(T, 0.0), val, 1e-6);
}

test "[loss] computeLoss MSE known error" {
    const testing = std.testing;
    const nOut: usize = params.nNeurons[params.nNeurons.len - 1];

    var pred: [nOut]T = undefined;
    var out: [nOut]T = undefined;
    var dL: [nOut]T = undefined;

    for (&pred, &out) |*p, *o| {
        p.* = 3.0;
        o.* = 1.0;
    }

    const lossVal = try computeLoss(&pred, &out, &dL, params.loss.MSE);
    const expectedLoss: T = 0.5 * 4.0;

    try testing.expectApproxEqAbs(expectedLoss, lossVal, 1e-4);
    for (dL) |val| try testing.expectApproxEqAbs(@as(T, 1.0), val, 1e-4);
}
