//! This module define the loss functions and their derivatives

// Imports the modules used
const std = @import("std");
const params = @import("params");
const errors = @import("errors");
const err = errors.lossError;
const fType = params.floatType;

// Mean square error loss and its derivatives
fn MSE(pred: []const fType, out: []const fType, dL: []fType, value: *fType) void {
    var loss: fType = 0.0;

    // Compute the loss and its derivative
    for (pred, out, dL) |*p, *o, *dl| {
        dl.* = p.* - o.*;
        loss += 0.5 * (dl.*) * (dl.*);
    }

    value.* = loss / @as(fType, @floatFromInt(out.len));
}

/// Function that computes the loss and its derivatives for a given choice of loss function
pub fn computeLoss(pred: []const fType, out: []const fType, dL: []fType, lossFunc: params.loss) !fType {
    // Compute the chunck size
    const chunkSize: usize = (pred.len + params.numThreads - 1) / params.numThreads;

    // Define the array of threads
    var threads: [params.numThreads]std.Thread = undefined;
    var lossThread: [params.numThreads]fType = undefined;
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

    // Compute the total final loss
    var totalLoss: fType = 0.0;
    for (&lossThread) |*lt| {
        totalLoss += lt.*;
    }

    return totalLoss;
}

test "[loss] computeLoss MSE zero error" {
    const testing = std.testing;
    const nOut: usize = params.nNeurons[params.nNeurons.len - 1];

    var pred: [nOut]fType = undefined;
    var out: [nOut]fType = undefined;
    var dL: [nOut]fType = undefined;

    for (&pred, &out) |*p, *o| {
        p.* = 1.0;
        o.* = 1.0;
    }

    const lossVal = try computeLoss(&pred, &out, &dL, params.loss.MSE);

    try testing.expectApproxEqAbs(@as(fType, 0.0), lossVal, 1e-6);
    for (dL) |val| try testing.expectApproxEqAbs(@as(fType, 0.0), val, 1e-6);
}

test "[loss] computeLoss MSE known error" {
    const testing = std.testing;
    const nOut: usize = params.nNeurons[params.nNeurons.len - 1];

    var pred: [nOut]fType = undefined;
    var out: [nOut]fType = undefined;
    var dL: [nOut]fType = undefined;

    for (&pred, &out) |*p, *o| {
        p.* = 3.0;
        o.* = 1.0;
    }

    const lossVal = try computeLoss(&pred, &out, &dL, params.loss.MSE);
    const expectedLoss: fType = 0.5 * 4.0;

    try testing.expectApproxEqAbs(expectedLoss, lossVal, 1e-4);
    for (dL) |val| try testing.expectApproxEqAbs(@as(fType, 2.0), val, 1e-4);
}
