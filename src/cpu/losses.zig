//! This module defines the loss functions and their derivatives.
//! The element-wise math is delegated to the Eigen C++ kernels in
//! src/eigen/losses.cpp; this module only dispatches based on the
//! configured loss function.

// Imports the modules used
const std = @import("std");
const params = @import("params");
const eigen = @import("eigen");
const T = params.T;

/// Computes the loss and its derivative for the given loss function.
/// Returns the scalar loss value and writes the derivative into `dL`.
/// Works for any array length (a single data point's output vector).
pub fn computeLoss(pred: []const T, out: []const T, dL: []T, lossFunc: params.loss) T {
    switch (lossFunc) {
        params.loss.MSE => return eigen.computeMSE(pred, out, dL),
    }
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

    const lossVal = computeLoss(&pred, &out, &dL, params.loss.MSE);

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

    const lossVal = computeLoss(&pred, &out, &dL, params.loss.MSE);
    const expectedLoss: T = 0.5 * 4.0;

    try testing.expectApproxEqAbs(expectedLoss, lossVal, 1e-4);
    for (dL) |val| try testing.expectApproxEqAbs(@as(T, 1.0), val, 1e-4);
}
