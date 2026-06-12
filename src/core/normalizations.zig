//! This module is used to normalize and denormalize the data
//! The data is normalized by x' = (x - b)/a => x = x'*a + b

// Import the modules used
const std = @import("std");
const params = @import("params");
const err = @import("errors").normalizationError;
const fType = params.floatType;

/// Computes Z-score normalization factors: `a` is the standard deviation and `b` is the mean
fn computeMeanStd(norm: *const Norm, inputs: []const fType, outputs: []const fType) void {
    const nIn = params.nNeurons[0];
    const nOut = params.nNeurons[params.nNeurons.len - 1];
    const nData: usize = inputs.len / nIn;
    const N: fType = @as(fType, @floatFromInt(nData));

    // Compute means
    for (0..nData) |i| {
        for (0..nIn) |j| {
            norm.bIn[j] += inputs[i * nIn + j] / N;
        }
        for (0..nOut) |j| {
            norm.bOut[j] += outputs[i * nOut + j] / N;
        }
    }

    // Compute the stds
    for (0..nData) |i| {
        for (0..nIn) |j| {
            norm.aIn[j] += (inputs[i * nIn + j] - norm.bIn[j]) * (inputs[i * nIn + j] - norm.bIn[j]) / N;
        }
        for (0..nOut) |j| {
            norm.aOut[j] += (outputs[i * nOut + j] - norm.bOut[j]) * (outputs[i * nOut + j] - norm.bOut[j]) / N;
        }
    }
    for (norm.aIn) |*a| a.* = std.math.sqrt(a.*);
    for (norm.aOut) |*a| a.* = std.math.sqrt(a.*);
}

/// Normalizes a chunk of data in-place using the stored factors: x' = (x - b) / a
fn normalizeChunk(norm: *const Norm, inputs: []fType, outputs: []fType, startData: usize, endData: usize) void {
    const nIn = params.nNeurons[0];
    const nOut = params.nNeurons[params.nNeurons.len - 1];

    // Normalize the inputs and outputs
    for (startData..endData) |i| {
        for (0..nIn) |j| {
            inputs[i * nIn + j] = (inputs[i * nIn + j] - norm.bIn[j]) / norm.aIn[j];
        }
        for (0..nOut) |j| {
            outputs[i * nOut + j] = (outputs[i * nOut + j] - norm.bOut[j]) / norm.aOut[j];
        }
    }
}

/// Denormalizes a chunk of data in-place using the stored factors: x = x' * a + b
fn denormalizeChunk(norm: *const Norm, inputs: []fType, outputs: []fType, startData: usize, endData: usize) void {
    const nIn = params.nNeurons[0];
    const nOut = params.nNeurons[params.nNeurons.len - 1];

    // Denormalize the inputs and outputs
    for (startData..endData) |i| {
        for (0..nIn) |j| {
            inputs[i * nIn + j] = (inputs[i * nIn + j]) * (norm.aIn[j]) + norm.bIn[j];
        }
        for (0..nOut) |j| {
            outputs[i * nOut + j] = (outputs[i * nOut + j]) * norm.aOut[j] + norm.bOut[j];
        }
    }
}

/// Main structure to handle the normalization and denormalization
pub const Norm = struct {
    allocator: std.mem.Allocator,
    aIn: []fType = &.{},
    bIn: []fType = &.{},
    aOut: []fType = &.{},
    bOut: []fType = &.{},

    /// Initializes the structure computing the mean and std of the inputs and outputs
    pub fn init(allocator: std.mem.Allocator) !Norm {
        // Create the structure to be outputed
        var norm = Norm{
            .allocator = allocator,
        };

        // Get the size of the inputs and outputs
        const nInputs = params.nNeurons[0];
        const nOutputs = params.nNeurons[params.nNeurons.len - 1];

        // Alloc the slices for the mean and std of inputs and outputs
        if (norm.allocator.alloc(fType, nInputs)) |slice| {
            norm.aIn = slice;
        } else |_| {
            std.log.err("Problem in the allocation of the mean of the inputs!\n", .{});
            return err.aAllocation;
        }

        // Allocate the slice for the std of the input
        if (norm.allocator.alloc(fType, nInputs)) |slice| {
            norm.bIn = slice;
        } else |_| {
            std.log.err("Problem in the allocation of the std of the inputs!\n", .{});
            return err.bAllocation;
        }

        // Zero the input arrays
        for (norm.aIn, norm.bIn) |*a, *b| {
            a.* = 0.0;
            b.* = 0.0;
        }

        // Allocate the slice for the mean of the outputs
        if (norm.allocator.alloc(fType, nOutputs)) |slice| {
            norm.aOut = slice;
        } else |_| {
            std.log.err("Problem in the allocation of the mean of the outputs!\n", .{});
            return err.aAllocation;
        }

        // Allocate the slice for the std of the outputs
        if (norm.allocator.alloc(fType, nOutputs)) |slice| {
            norm.bOut = slice;
        } else |_| {
            std.log.err("Problem in the allocation of the std of the outputs!\n", .{});
            return err.bAllocation;
        }

        // Zero the outputs arrys
        for (norm.aOut, norm.bOut) |*a, *b| {
            a.* = 0.0;
            b.* = 0.0;
        }

        return norm;
    }

    test "[norms] Norm init and deinit" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        var norm = try Norm.init(allocator);
        defer norm.deinit();

        const nIn: usize = params.nNeurons[0];
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];

        try std.testing.expectEqual(@as(usize, nIn), norm.aIn.len);
        try std.testing.expectEqual(@as(usize, nIn), norm.bIn.len);
        try std.testing.expectEqual(@as(usize, nOut), norm.aOut.len);
        try std.testing.expectEqual(@as(usize, nOut), norm.bOut.len);

        for (norm.aIn) |val| try std.testing.expectEqual(@as(fType, 0.0), val);
        for (norm.bIn) |val| try std.testing.expectEqual(@as(fType, 0.0), val);
        for (norm.aOut) |val| try std.testing.expectEqual(@as(fType, 0.0), val);
        for (norm.bOut) |val| try std.testing.expectEqual(@as(fType, 0.0), val);
    }

    /// Frees all memory allocated in the structure
    pub fn deinit(self: *const Norm) void {
        self.allocator.free(self.aIn);
        self.allocator.free(self.bIn);
        self.allocator.free(self.aOut);
        self.allocator.free(self.bOut);
    }

    /// Computes the normalization factors
    pub fn computeNormalization(self: *const Norm, inputs: []const fType, outputs: []const fType) !void {
        const nIn = params.nNeurons[0];
        const nOut = params.nNeurons[params.nNeurons.len - 1];

        // Check if the size of inputs and outputs are compatible
        if (inputs.len / nIn != outputs.len / nOut) {
            std.log.err("Inputs and outputs must have the same number of data vectors!\n", .{});
            return err.incompatibleSizes;
        }

        // Check if the normalizations were computed
        if (self.aIn.len == 0) {
            std.log.err("The normalization structure was not initialized!\n", .{});
            return err.notInitialized;
        }

        // Compute the slope and shift for the inputs and outputs
        computeMeanStd(self, inputs, outputs);
    }

    test "[norms] computeNormalization" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        var norm = try Norm.init(allocator);
        defer norm.deinit();

        var inputs = [_]fType{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
        var outputs = [_]fType{ 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0 };

        try norm.computeNormalization(&inputs, &outputs);

        for (norm.aIn) |val| try std.testing.expect(val > 0);
        for (norm.aOut) |val| try std.testing.expect(val > 0);
    }

    /// Normalizes the inputs and outputs given
    pub fn normalize(self: *const Norm, inputs: []fType, outputs: []fType) !void {
        const nIn = params.nNeurons[0];
        const nOut = params.nNeurons[params.nNeurons.len - 1];

        // Check if the size of inputs and outputs are compatible
        if (inputs.len / nIn != outputs.len / nOut) {
            std.log.err("Inputs and outputs must have the same number of data vectors!\n", .{});
            return err.incompatibleSizes;
        }

        // Check if the normalizations were computed
        if (self.aIn.len == 0) {
            std.log.err("The normalization structure was not initialized!\n", .{});
            return err.notInitialized;
        }

        // Compute the number of data points
        const nData: usize = inputs.len / nIn;

        // Compute the chunck size
        const chunkSize: usize = (nData + params.numThreads - 1) / params.numThreads;

        // Define the array of threads
        var threads: [params.numThreads]std.Thread = undefined;

        // Compute the chuncks for each thread
        for (&threads, 0..params.numThreads) |*thread, i| {
            const start: usize = i * chunkSize;
            const end: usize = @min(start + chunkSize, nData);

            // Normalize the inputs and outputs in the current chunk
            if (std.Thread.spawn(.{}, normalizeChunk, .{ self, inputs, outputs, start, end })) |t| {
                thread.* = t;
            } else |_| {
                std.log.err("Error in spawning thread {}!\n", .{i});
                return err.threadRun;
            }
        }

        // Await all threads to finish
        for (&threads) |*thread| {
            thread.*.join();
        }
    }

    test "[norms] normalize" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        var norm = try Norm.init(allocator);
        defer norm.deinit();

        const nIn: usize = params.nNeurons[0];
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];
        const nData: usize = 5;

        var inputs = [_]fType{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
        var outputs = [_]fType{ 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0 };

        try norm.computeNormalization(&inputs, &outputs);
        try norm.normalize(&inputs, &outputs);

        var sumIn: fType = 0;
        var sumOut: fType = 0;
        for (0..nData) |i| {
            for (0..nIn) |j| {
                sumIn += inputs[i * nIn + j];
            }
            for (0..nOut) |j| {
                sumOut += outputs[i * nOut + j];
            }
        }
        const meanIn = sumIn / @as(fType, @floatFromInt(nData * nIn));
        const meanOut = sumOut / @as(fType, @floatFromInt(nData * nOut));

        const tol: fType = 1e-4;
        try std.testing.expectApproxEqAbs(@as(fType, 0.0), meanIn, tol);
        try std.testing.expectApproxEqAbs(@as(fType, 0.0), meanOut, tol);
    }

    /// Denormalize the inputs and outputs
    pub fn denormalize(self: *const Norm, inputs: []fType, outputs: []fType) !void {
        const nIn = params.nNeurons[0];
        const nOut = params.nNeurons[params.nNeurons.len - 1];

        // Check if the size of inputs and outputs are compatible
        if (inputs.len / nIn != outputs.len / nOut) {
            std.log.err("Inputs and outputs must have the same number of data vectors!\n", .{});
            return err.incompatibleSizes;
        }

        // Check if the normalizations were computed
        if (self.bIn.len == 0) {
            std.log.err("The normalization structure was not initialized!\n", .{});
            return err.notInitialized;
        }

        // Compute the number of data points
        const nData: usize = inputs.len / nIn;

        // Compute the chunck size
        const chunkSize: usize = (nData + params.numThreads - 1) / params.numThreads;

        // Define the array of threads
        var threads: [params.numThreads]std.Thread = undefined;

        // Compute the chuncks for each thread
        for (&threads, 0..params.numThreads) |*thread, i| {
            const start: usize = i * chunkSize;
            const end: usize = @min(start + chunkSize, nData);

            // Denormalize the inputs and outputs in the current chunk
            if (std.Thread.spawn(.{}, denormalizeChunk, .{ self, inputs, outputs, start, end })) |t| {
                thread.* = t;
            } else |_| {
                std.log.err("Error in spawning thread {}!\n", .{i});
                return err.threadRun;
            }
        }

        // Await all threads to finish
        for (&threads) |*thread| {
            thread.*.join();
        }
    }

    test "[norms] denormalize" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        var norm = try Norm.init(allocator);
        defer norm.deinit();

        var inputs = [_]fType{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
        var outputs = [_]fType{ 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0 };

        const origIn = inputs;
        const origOut = outputs;

        try norm.computeNormalization(&inputs, &outputs);
        try norm.normalize(&inputs, &outputs);
        try norm.denormalize(&inputs, &outputs);

        const tol: fType = 1e-4;
        for (&inputs, &origIn) |*val, *orig| {
            try std.testing.expectApproxEqAbs(orig.*, val.*, tol);
        }
        for (&outputs, &origOut) |*val, *orig| {
            try std.testing.expectApproxEqAbs(orig.*, val.*, tol);
        }
    }

    test "[norms] normalize-denormalize round-trip with random data" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        var norm = try Norm.init(allocator);
        defer norm.deinit();

        const nIn: usize = params.nNeurons[0];
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];
        const nData: usize = 20;

        var xoshiro256 = std.Random.Xoshiro256.init(42);
        const rand = std.Random.Xoshiro256.random(&xoshiro256);

        var inputs: [nData * nIn]fType = undefined;
        var outputs: [nData * nOut]fType = undefined;

        for (&inputs) |*val| val.* = std.Random.float(rand, fType) * 10.0;
        for (&outputs) |*val| val.* = std.Random.float(rand, fType) * 10.0;

        const origIn = inputs;
        const origOut = outputs;

        try norm.computeNormalization(&inputs, &outputs);
        try norm.normalize(&inputs, &outputs);
        try norm.denormalize(&inputs, &outputs);

        const tol: fType = 1e-4;
        for (&inputs, &origIn) |*val, *orig| {
            try std.testing.expectApproxEqAbs(orig.*, val.*, tol);
        }
        for (&outputs, &origOut) |*val, *orig| {
            try std.testing.expectApproxEqAbs(orig.*, val.*, tol);
        }
    }
};

// Run the tests for the norm structure
comptime {
    std.testing.refAllDecls(@This());
}
