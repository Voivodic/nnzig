//! This module defines the activation functions and their derivatives.
//! The element-wise math is delegated to the Eigen C++ kernels in
//! src/eigen/activations.cpp; this module only dispatches based on the
//! configured activation function. Because the computation is element-wise,
//! a single call handles any array length, including a whole batch.

// Import the modules used
const std = @import("std");
const params = @import("params");
const eigen = @import("eigen");
const testing = std.testing;
const T = params.T;

/// Apply the activation function to each element of `input` (in place) and
/// compute its derivative into `df`. Works for any array length, so a whole
/// batch of data points can be processed in a single call.
pub fn activateElements(input: []T, df: []T, act: params.activation) void {
    switch (act) {
        params.activation.none => eigen.activateNone(input, df),
        params.activation.relu => eigen.activateRelu(input, df),
        params.activation.tanh => eigen.activateTanh(input, df),
        params.activation.sigmoid => eigen.activateSigmoid(input, df),
    }
}

test "[act] activateElements none" {
    // Create the fake input and derivative arrays
    var input = [_]T{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var df: [5]T = undefined;

    // Compute the activation
    activateElements(&input, &df, params.activation.none);

    // Check the results
    for (input) |val| try testing.expect(std.math.isFinite(val));
    for (df) |val| try testing.expectEqual(@as(T, 1.0), val);
}

test "[act] activateElements relu" {
    // Create the fake input and derivative arrays
    var input = [_]T{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var df: [5]T = undefined;

    // Compute the activation
    activateElements(&input, &df, params.activation.relu);

    // Check the results
    try testing.expectEqual(@as(T, 0.0), input[0]);
    try testing.expectEqual(@as(T, 0.0), input[1]);
    try testing.expectEqual(@as(T, 0.0), input[2]);
    try testing.expectEqual(@as(T, 1.0), input[3]);
    try testing.expectEqual(@as(T, 2.0), input[4]);

    try testing.expectEqual(@as(T, 0.0), df[0]);
    try testing.expectEqual(@as(T, 0.0), df[1]);
    try testing.expectEqual(@as(T, 1.0), df[2]);
    try testing.expectEqual(@as(T, 1.0), df[3]);
    try testing.expectEqual(@as(T, 1.0), df[4]);
}

test "[act] activateElements tanh" {
    // Create the fake input and derivative arrays
    var input = [_]T{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var df: [5]T = undefined;

    // Compute the activation
    activateElements(&input, &df, params.activation.tanh);

    // Check the results
    try testing.expectApproxEqAbs(@as(T, -0.964028), input[0], 1e-3);
    try testing.expectApproxEqAbs(@as(T, -0.761594), input[1], 1e-3);
    try testing.expectApproxEqAbs(@as(T, 0.0), input[2], 1e-3);
    try testing.expectApproxEqAbs(@as(T, 0.761594), input[3], 1e-3);
    try testing.expectApproxEqAbs(@as(T, 0.964028), input[4], 1e-3);

    try testing.expectApproxEqAbs(@as(T, 0.0706508), df[0], 1e-3);
    try testing.expectApproxEqAbs(@as(T, 0.419974), df[1], 1e-3);
    try testing.expectApproxEqAbs(@as(T, 1.0), df[2], 1e-3);
    try testing.expectApproxEqAbs(@as(T, 0.419974), df[3], 1e-3);
    try testing.expectApproxEqAbs(@as(T, 0.0706508), df[4], 1e-3);
}

test "[act] activateElements sigmoid" {
    // Create the fake input and derivative arrays
    var input = [_]T{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var df: [5]T = undefined;

    // Compute the activation
    activateElements(&input, &df, params.activation.sigmoid);

    // Check the results
    try testing.expectApproxEqAbs(@as(T, 0.119203), input[0], 1e-3);
    try testing.expectApproxEqAbs(@as(T, 0.268941), input[1], 1e-3);
    try testing.expectApproxEqAbs(@as(T, 0.5), input[2], 1e-3);
    try testing.expectApproxEqAbs(@as(T, 0.731059), input[3], 1e-3);
    try testing.expectApproxEqAbs(@as(T, 0.880797), input[4], 1e-3);

    try testing.expectApproxEqAbs(@as(T, 0.104994), df[0], 1e-3);
    try testing.expectApproxEqAbs(@as(T, 0.196612), df[1], 1e-3);
    try testing.expectApproxEqAbs(@as(T, 0.25), df[2], 1e-3);
    try testing.expectApproxEqAbs(@as(T, 0.196612), df[3], 1e-3);
    try testing.expectApproxEqAbs(@as(T, 0.104994), df[4], 1e-3);
}
