const std = @import("std");
const nnzig = @import("nnzig");

pub fn main() !void {
    // Define a allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) std.debug.print("Leak!\n", .{});
    }

    // Define the mlp
    var nn: nnzig.NN(f32) = try nnzig.NN(f32).init(allocator);
    defer nn.deinit();

    // Initialize the random generator
    var xoshiro256 = std.Random.Xoshiro256.init(12345);
    const rand = std.Random.Xoshiro256.random(&xoshiro256);

    // Create some fake data
    const Ndata: usize = 1000;
    const Nin: usize = 2;
    const Nout: usize = 2;
    var input: [][]f32 = undefined;
    var output: [][]f32 = undefined;

    input = try allocator.alloc([]f32, Ndata);
    output = try allocator.alloc([]f32, Ndata);
    for (input, output) |*in, *out| {
        in.* = try allocator.alloc(f32, Nin);
        out.* = try allocator.alloc(f32, Nout);

        for (in.*) |*i| {
            i.* = std.Random.floatNorm(rand, f32);
        }
        out.*[0] = 2.0 + 1.2 * in.*[0] - std.math.pow(f32, in.*[1], 2.3) + @exp(-3.0 * in.*[0] - 2.0 * in.*[1]);
        out.*[1] = 1.4 + 3.0 * in.*[0] - std.math.pow(f32, in.*[1], 1.7) + @exp(-2.0 * in.*[0] - 3.0 * in.*[1]);
    }
    defer {
        for (input, output) |*in, *out| {
            allocator.free(in.*);
            allocator.free(out.*);
        }
        allocator.free(input);
        allocator.free(output);
    }

    // Compute the NN for this input
    const output_pred = try nn.forward(input[0]);
    for (0..output_pred.len) |i| {
        std.debug.print("output[{}] = {}\n", .{ i, output_pred[i] });
    }

    // Compute the loss
    const loss = try nn.backProp(input, output);
    std.debug.print("Loss = {}\n", .{loss});
}
