const std = @import("std");

// Define the possible losses
pub const losses = enum(u8) {
    MSE,
};

// Define the mean square error loss
fn MSE(comptime T: type, pred: []const T, out: []const T, dL: []T) T {
    var loss: T = 0.0;

    // Compute the loss and its derivative
    for (pred, out, dL) |*p, *o, *dl| {
        dl.* = p.* - o.*;
        loss += 0.5 * (dl.*) * (dl.*);
    }

    return loss / @as(T, @floatFromInt(out.len));
}

// Function that computes the loss function
pub fn computeLoss(comptime T: type, pred: []const T, out: []const T, dL: []T, lossFunc: losses) T {
    // Check which loss function to use
    return switch (lossFunc) {
        losses.MSE => MSE(T, pred, out, dL),
    };
}
