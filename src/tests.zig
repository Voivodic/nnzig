const std = @import("std");
const nnzig = @import("nnzig");
const params = @import("params");
const testing = std.testing;

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
    var input = try allocator.alloc(f32, Ndata * Nin);
    var output = try allocator.alloc(f32, Ndata * Nout);
    defer {
        allocator.free(input);
        allocator.free(output);
    }

    for (0..Ndata) |i| {
        for (0..Nin) |j| {
            input[i * Nin + j] = std.Random.floatNorm(rand, f32);
        }
        output[i * Nout] = 2.0 + 1.2 * input[i * Nin] - std.math.pow(f32, input[i * Nin + 1], 2) + @exp(-3.0 * input[i * Nin] - 2.0 * input[i * Nin + 1]);
        output[i * Nout + 1] = 1.4 + 3.0 * input[i * Nin] - std.math.pow(f32, input[i * Nin + 1], 2) + @exp(-2.0 * input[i * Nin] - 3.0 * input[i * Nin + 1]);
    }

    try nn.computeNormalization(input, output);
    try nn.normalize(input, output);
}

test "NN init correct field sizes" {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) std.debug.print("Leak!\n", .{});
    }

    const ioContext = std.testing.io;

    var nn: nnzig.NN = try nnzig.NN.init(allocator, ioContext);
    defer nn.deinit();

    var nWeights: usize = 0;
    var nBiases: usize = 0;
    for (1..params.nNeurons.len) |i| {
        nWeights += params.nNeurons[i] * params.nNeurons[i - 1];
        nBiases += params.nNeurons[i];
    }

    try testing.expectEqual(nWeights, nn.weights.len);
    try testing.expectEqual(nWeights, nn.gradW.len);
    try testing.expectEqual(nWeights, nn.mW.len);
    try testing.expectEqual(nWeights, nn.vW.len);
    try testing.expectEqual(nBiases, nn.biases.len);
    try testing.expectEqual(nBiases, nn.gradB.len);
    try testing.expectEqual(nBiases, nn.mB.len);
    try testing.expectEqual(nBiases, nn.vB.len);
}

test "NN init adam arrays zeroed" {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) std.debug.print("Leak!\n", .{});
    }

    const ioContext = std.testing.io;

    var nn: nnzig.NN = try nnzig.NN.init(allocator, ioContext);
    defer nn.deinit();

    for (nn.mW) |val| try testing.expectEqual(@as(f32, 0.0), val);
    for (nn.vW) |val| try testing.expectEqual(@as(f32, 0.0), val);
    for (nn.mB) |val| try testing.expectEqual(@as(f32, 0.0), val);
    for (nn.vB) |val| try testing.expectEqual(@as(f32, 0.0), val);
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

    const input = [_]f32{ 1.0, 2.0 };
    const output = try nn.forward(&input);

    const nOut = params.nNeurons[params.nNeurons.len - 1];
    try testing.expectEqual(nOut, output.len);
}

test "NN zeroGrad zeros all gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) std.debug.print("Leak!\n", .{});
    }

    const ioContext = std.testing.io;

    var nn: nnzig.NN = try nnzig.NN.init(allocator, ioContext);
    defer nn.deinit();

    @memset(nn.gradW, 3.14);
    @memset(nn.gradB, 2.71);

    nn.zeroGrad();

    for (nn.gradW) |val| try testing.expectEqual(@as(f32, 0.0), val);
    for (nn.gradB) |val| try testing.expectEqual(@as(f32, 0.0), val);
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
    var input = try allocator.alloc(f32, Ndata * Nin);
    var output = try allocator.alloc(f32, Ndata * Nout);
    defer {
        allocator.free(input);
        allocator.free(output);
    }

    for (0..Ndata) |i| {
        for (0..Nin) |j| {
            input[i * Nin + j] = std.Random.floatNorm(rand, f32);
        }
        output[i * Nout] = 2.0 + 1.2 * input[i * Nin] - std.math.pow(f32, input[i * Nin + 1], 2) + @exp(-3.0 * input[i * Nin] - 2.0 * input[i * Nin + 1]);
        output[i * Nout + 1] = 1.4 + 3.0 * input[i * Nin] - std.math.pow(f32, input[i * Nin + 1], 2) + @exp(-2.0 * input[i * Nin] - 3.0 * input[i * Nin + 1]);
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
    var input = try allocator.alloc(f32, Ndata * Nin);
    var output = try allocator.alloc(f32, Ndata * Nout);
    defer {
        allocator.free(input);
        allocator.free(output);
    }

    for (0..Ndata) |i| {
        for (0..Nin) |j| {
            input[i * Nin + j] = std.Random.floatNorm(rand, f32);
        }
        output[i * Nout] = 2.0 + 1.2 * input[i * Nin] - std.math.pow(f32, input[i * Nin + 1], 2) + @exp(-3.0 * input[i * Nin] - 2.0 * input[i * Nin + 1]);
        output[i * Nout + 1] = 1.4 + 3.0 * input[i * Nin] - std.math.pow(f32, input[i * Nin + 1], 2) + @exp(-2.0 * input[i * Nin] - 3.0 * input[i * Nin + 1]);
    }

    try nn.computeNormalization(input, output);
    try nn.normalize(input, output);
    try nn.train(input, output);

    try testing.expectEqual(@as(usize, params.nEpochs), nn.lossesTraining.len);
    try testing.expectEqual(@as(usize, params.nEpochs), nn.lossesValidation.len);
    try testing.expect(nn.lossesValidation[0] > nn.lossesValidation[params.nEpochs - 1]);
}
