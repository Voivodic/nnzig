//! This module define the loss functions and their derivatives

// Imports the modules used
const std = @import("std");
const params = @import("params");
const errors = @import("errors");
const err = errors.lossError;

// Mean square error loss and its derivatives
fn MSE(comptime T: type, pred: []const T, out: []const T, dL: []T, value: *T) void {
    var loss: T = 0.0;

    // Compute the loss and its derivative
    for (pred, out, dL) |*p, *o, *dl| {
        dl.* = p.* - o.*;
        loss += 0.5 * (dl.*) * (dl.*);
    }

    value.* = loss / @as(T, @floatFromInt(out.len));
}

/// Function that computes the loss and its derivatives for a given choice of loss function
pub fn computeLoss(comptime T: type, pred: []const T, out: []const T, dL: []T, lossFunc: params.loss) !T {
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
                if (std.Thread.spawn(.{}, MSE, .{ T, pred[start..end], out[start..end], dL[start..end], lt })) |t| {
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
    var totalLoss: T = 0.0;
    for (&lossThread) |*lt| {
        totalLoss += lt.*;
    }

    return totalLoss;
}
