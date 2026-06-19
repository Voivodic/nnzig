//! Integration tests for the nnzig library, exercising the full neural network
//! lifecycle: initialization, normalization, training, and weight persistence.

const std = @import("std");
const nnzig = @import("nnzig");
const params = @import("params");
const testing = std.testing;

const T: type = params.T;

fn pow(x: T, n: T) T {
    return @floatCast(std.math.pow(f64, @floatCast(x), @floatCast(n)));
}

test "NN init and deinit" {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) std.debug.print("Leak!\n", .{});
    }

    const ioContext = std.testing.io;

    var nn: nnzig.NN = try nnzig.NN.init(allocator, ioContext);
    defer nn.deinit();
}

test "NN compute normalization and normalize" {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) std.debug.print("Leak!\n", .{});
    }

    const ioContext = std.testing.io;

    var nn: nnzig.NN = try nnzig.NN.init(allocator, ioContext);
    defer nn.deinit();

    var xoshiro256 = std.Random.Xoshiro256.init(12345);
    const rand = std.Random.Xoshiro256.random(&xoshiro256);

    const Ndata: usize = 100;
    const Nin: usize = 2;
    const Nout: usize = 2;
    var input = try allocator.alloc(T, Ndata * Nin);
    var output = try allocator.alloc(T, Ndata * Nout);
    defer {
        allocator.free(input);
        allocator.free(output);
    }

    for (0..Ndata) |i| {
        for (0..Nin) |j| {
            input[i * Nin + j] = @floatCast(std.Random.floatNorm(rand, f32));
        }
        const x0: T = @as(T, input[i * Nin]);
        const x1: T = @as(T, input[i * Nin + 1]);
        output[i * Nout] = @floatCast(2.0 + 1.2 * x0 - pow(x1, 2) + @exp(-3.0 * x0 - 2.0 * x1));
        output[i * Nout + 1] = @floatCast(1.4 + 3.0 * x0 - pow(x1, 2) + @exp(-2.0 * x0 - 3.0 * x1));
    }

    try nn.computeNormalization(input, output);
    try nn.normalize(input, output);
}

test "NN forward pass output size" {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) std.debug.print("Leak!\n", .{});
    }

    const ioContext = std.testing.io;

    var nn: nnzig.NN = try nnzig.NN.init(allocator, ioContext);
    defer nn.deinit();

    const input = [_]T{ 1.0, 2.0 };
    const output = try nn.forward(&input);

    const nOut = params.nNeurons[params.nNeurons.len - 1];
    try testing.expectEqual(nOut, output.len);
}

test "NN normalization factors positive std" {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) std.debug.print("Leak!\n", .{});
    }

    const ioContext = std.testing.io;

    var nn: nnzig.NN = try nnzig.NN.init(allocator, ioContext);
    defer nn.deinit();

    var xoshiro256 = std.Random.Xoshiro256.init(12345);
    const rand = std.Random.Xoshiro256.random(&xoshiro256);

    const Ndata: usize = 100;
    const Nin: usize = 2;
    const Nout: usize = 2;
    var input = try allocator.alloc(T, Ndata * Nin);
    var output = try allocator.alloc(T, Ndata * Nout);
    defer {
        allocator.free(input);
        allocator.free(output);
    }

    for (0..Ndata) |i| {
        for (0..Nin) |j| {
            input[i * Nin + j] = @floatCast(std.Random.floatNorm(rand, f32));
        }
        const x0: T = @as(T, input[i * Nin]);
        const x1: T = @as(T, input[i * Nin + 1]);
        output[i * Nout] = @floatCast(2.0 + 1.2 * x0 - pow(x1, 2) + @exp(-3.0 * x0 - 2.0 * x1));
        output[i * Nout + 1] = @floatCast(1.4 + 3.0 * x0 - pow(x1, 2) + @exp(-2.0 * x0 - 3.0 * x1));
    }

    try nn.computeNormalization(input, output);

    try testing.expectEqual(@as(usize, Nin), nn.norm.aIn.len);
    try testing.expectEqual(@as(usize, Nin), nn.norm.bIn.len);
    try testing.expectEqual(@as(usize, Nout), nn.norm.aOut.len);
    try testing.expectEqual(@as(usize, Nout), nn.norm.bOut.len);

    for (nn.norm.aIn) |val| try testing.expect(val > 0.0);
    for (nn.norm.aOut) |val| try testing.expect(val > 0.0);
}

test "NN training" {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) std.debug.print("Leak!\n", .{});
    }

    const ioContext = std.testing.io;

    var nn: nnzig.NN = try nnzig.NN.init(allocator, ioContext);
    defer nn.deinit();

    var xoshiro256 = std.Random.Xoshiro256.init(12345);
    const rand = std.Random.Xoshiro256.random(&xoshiro256);

    const Ndata: usize = 100;
    const Nin: usize = 2;
    const Nout: usize = 2;
    var input = try allocator.alloc(T, Ndata * Nin);
    var output = try allocator.alloc(T, Ndata * Nout);
    defer {
        allocator.free(input);
        allocator.free(output);
    }

    for (0..Ndata) |i| {
        for (0..Nin) |j| {
            input[i * Nin + j] = @floatCast(std.Random.floatNorm(rand, f32));
        }
        const x0: T = @as(T, input[i * Nin]);
        const x1: T = @as(T, input[i * Nin + 1]);
        output[i * Nout] = @floatCast(2.0 + 1.2 * x0 - pow(x1, 2) + @exp(-3.0 * x0 - 2.0 * x1));
        output[i * Nout + 1] = @floatCast(1.4 + 3.0 * x0 - pow(x1, 2) + @exp(-2.0 * x0 - 3.0 * x1));
    }

    try nn.computeNormalization(input, output);
    try nn.normalize(input, output);
    try nn.train(input, output);

    try testing.expectEqual(@as(usize, params.nEpochs), nn.lossesTraining.len);
    try testing.expectEqual(@as(usize, params.nEpochs), nn.lossesValidation.len);
    try testing.expect(nn.lossesValidation[0] > nn.lossesValidation[params.nEpochs - 1]);
}
