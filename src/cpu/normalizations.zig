//! This module is used to normalize and denormalize the data.
//! The data is normalized by x' = (x - b)/a => x = x'*a + b.
//! The factor computation and the (de)normalization transforms are delegated to
//! the Eigen C++ kernels in src/eigen/normalizations.cpp; this module owns the
//! Norm structure (the factor slices) and validates the inputs.

// Import the modules used
const std = @import("std");
const params = @import("params");
const eigen = @import("eigen");
const err = @import("errors").normalizationError;
const T = params.T;

/// Holds the Z-score scale (`a*`) and shift (`b*`) factors for both inputs and outputs, used to normalize and denormalize the data via x' = (x - b)/a.
pub const Norm = struct {
    allocator: std.mem.Allocator,
    aIn: []T = &.{},
    bIn: []T = &.{},
    aOut: []T = &.{},
    bOut: []T = &.{},

    /// Initializes the structure by allocating and zeroing the scale and shift slices for the inputs and outputs.
    pub fn init(allocator: std.mem.Allocator) !Norm {
        // Create the structure to be outputed
        var norm = Norm{
            .allocator = allocator,
        };

        // Get the size of the inputs and outputs
        const nInputs = params.nNeurons[0];
        const nOutputs = params.nNeurons[params.nNeurons.len - 1];

        // Alloc the slices for the mean and std of inputs and outputs
        if (norm.allocator.alloc(T, nInputs)) |slice| {
            norm.aIn = slice;
        } else |_| {
            std.log.err("Problem in the allocation of the mean of the inputs!\n", .{});
            return err.aAllocation;
        }

        // Allocate the slice for the std of the input
        if (norm.allocator.alloc(T, nInputs)) |slice| {
            norm.bIn = slice;
        } else |_| {
            std.log.err("Problem in the allocation of the std of the inputs!\n", .{});
            return err.bAllocation;
        }

        // Zero the input arrays
        eigen.setZero(norm.aIn);
        eigen.setZero(norm.bIn);

        // Allocate the slice for the mean of the outputs
        if (norm.allocator.alloc(T, nOutputs)) |slice| {
            norm.aOut = slice;
        } else |_| {
            std.log.err("Problem in the allocation of the mean of the outputs!\n", .{});
            return err.aAllocation;
        }

        // Allocate the slice for the std of the outputs
        if (norm.allocator.alloc(T, nOutputs)) |slice| {
            norm.bOut = slice;
        } else |_| {
            std.log.err("Problem in the allocation of the std of the outputs!\n", .{});
            return err.bAllocation;
        }

        // Zero the outputs arrys
        eigen.setZero(norm.aOut);
        eigen.setZero(norm.bOut);

        return norm;
    }

    /// Frees all memory allocated in the structure.
    pub fn deinit(self: *const Norm) void {
        self.allocator.free(self.aIn);
        self.allocator.free(self.bIn);
        self.allocator.free(self.aOut);
        self.allocator.free(self.bOut);
    }

    test "[norms] Norm init and deinit" {
        // Create the allocator
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        // Initialize the structure
        var norm = try Norm.init(allocator);
        defer norm.deinit();

        // Get the size of the inputs and outputs
        const nIn: usize = params.nNeurons[0];
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];

        // Check the size of the arrays
        try std.testing.expectEqual(nIn, norm.aIn.len);
        try std.testing.expectEqual(nIn, norm.bIn.len);
        try std.testing.expectEqual(nOut, norm.aOut.len);
        try std.testing.expectEqual(nOut, norm.bOut.len);

        // Check the values of the arrays
        for (norm.aIn) |val| try std.testing.expectEqual(@as(T, 0.0), val);
        for (norm.bIn) |val| try std.testing.expectEqual(@as(T, 0.0), val);
        for (norm.aOut) |val| try std.testing.expectEqual(@as(T, 0.0), val);
        for (norm.bOut) |val| try std.testing.expectEqual(@as(T, 0.0), val);
    }

    /// Computes the per-feature mean (stored in `bIn`/`bOut`) and population standard deviation (stored in `aIn`/`aOut`) from the given data. Returns `incompatibleSizes` if the input and output counts differ, or `notInitialized` if `init` was not called.
    pub fn computeNormalization(self: *const Norm, inputs: []const T, outputs: []const T) !void {
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
        eigen.computeMeanStd(inputs, outputs, self.aIn, self.bIn, self.aOut, self.bOut);
    }

    test "[norms] computeNormalization" {
        // Create the allocator
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        // Initialize the structure
        var norm = try Norm.init(allocator);
        defer norm.deinit();

        // Create the inputs and outputs
        var inputs = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
        var outputs = [_]T{ 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0 };

        // Compute the normalization factors
        try norm.computeNormalization(&inputs, &outputs);

        // Check the values of the arrays
        // feature 0: {1,3,5,7,9}  mean=5.0    std=sqrt(8)≈2.828427
        // feature 1: {2,4,6,8,10} mean=6.0    std=sqrt(8)≈2.828427
        // output feature 0: {2,6,10,14,18}  mean=10.0 std=sqrt(32)≈5.656854
        // output feature 1: {4,8,12,16,20}  mean=12.0 std=sqrt(32)≈5.656854
        try std.testing.expectApproxEqAbs(@as(T, 5.0), norm.bIn[0], 1e-3);
        try std.testing.expectApproxEqAbs(@as(T, 6.0), norm.bIn[1], 1e-3);
        try std.testing.expectApproxEqAbs(@as(T, 2.828427), norm.aIn[0], 1e-3);
        try std.testing.expectApproxEqAbs(@as(T, 2.828427), norm.aIn[1], 1e-3);
        try std.testing.expectApproxEqAbs(@as(T, 10.0), norm.bOut[0], 1e-3);
        try std.testing.expectApproxEqAbs(@as(T, 12.0), norm.bOut[1], 1e-3);
        try std.testing.expectApproxEqAbs(@as(T, 5.656854), norm.aOut[0], 1e-3);
        try std.testing.expectApproxEqAbs(@as(T, 5.656854), norm.aOut[1], 1e-3);
    }

    /// Normalizes the given inputs and outputs in place by applying x' = (x - b)/a with the previously computed factors. Requires `computeNormalization` to have run first.
    pub fn normalize(self: *const Norm, inputs: []T, outputs: []T) !void {
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

        // Normalize the inputs and outputs in place using Eigen
        eigen.normalize(inputs, outputs, self.aIn, self.bIn, self.aOut, self.bOut);
    }

    test "[norms] normalize" {
        // Create the allocator
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        // Initialize the structure
        var norm = try Norm.init(allocator);
        defer norm.deinit();

        // Get the size of the inputs and outputs
        const nIn: usize = 2;
        const nOut: usize = 2;
        const nData: usize = 5;

        // Create the inputs and outputs
        var inputs = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
        var outputs = [_]T{ 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0 };

        // Compute the normalization factors and normalize the inputs and outputs
        try norm.computeNormalization(&inputs, &outputs);
        try norm.normalize(&inputs, &outputs);

        // Compute the mean
        var sumIn: T = 0;
        var sumOut: T = 0;
        for (0..nData) |i| {
            for (0..nIn) |j| {
                sumIn += inputs[i * nIn + j];
            }
            for (0..nOut) |j| {
                sumOut += outputs[i * nOut + j];
            }
        }
        const meanIn = sumIn / @as(T, @floatFromInt(nData * nIn));
        const meanOut = sumOut / @as(T, @floatFromInt(nData * nOut));

        // Check the values of the arrays
        try std.testing.expectApproxEqAbs(@as(T, 0.0), meanIn, 1e-3);
        try std.testing.expectApproxEqAbs(@as(T, 0.0), meanOut, 1e-3);

        // Check that the std of each feature is approximately 1
        for (0..nIn) |j| {
            var sumSq: T = 0.0;
            for (0..nData) |i| {
                const v = inputs[i * nIn + j];
                sumSq += v * v;
            }
            const varIn = sumSq / @as(T, @floatFromInt(nData));
            try std.testing.expectApproxEqAbs(@as(T, 1.0), @sqrt(varIn), 1e-3);
        }
        for (0..nOut) |j| {
            var sumSq: T = 0.0;
            for (0..nData) |i| {
                const v = outputs[i * nOut + j];
                sumSq += v * v;
            }
            const varOut = sumSq / @as(T, @floatFromInt(nData));
            try std.testing.expectApproxEqAbs(@as(T, 1.0), @sqrt(varOut), 1e-3);
        }
    }

    /// Denormalizes the given inputs and outputs in place by reversing the transform, x = x'*a + b. Requires `computeNormalization` to have run first.
    pub fn denormalize(self: *const Norm, inputs: []T, outputs: []T) !void {
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

        // Denormalize the inputs and outputs in place using Eigen
        eigen.denormalize(inputs, outputs, self.aIn, self.bIn, self.aOut, self.bOut);
    }

    test "[norms] denormalize" {
        // Create the allocator
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        // Initialize the structure
        var norm = try Norm.init(allocator);
        defer norm.deinit();

        // Create the inputs and outputs
        var inputs = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
        var outputs = [_]T{ 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0 };

        // Save a copy of the original values
        const origIn = inputs;
        const origOut = outputs;

        // Normalize and denormalize the inputs and outputs
        try norm.computeNormalization(&inputs, &outputs);
        try norm.normalize(&inputs, &outputs);
        try norm.denormalize(&inputs, &outputs);

        // Check if the values are back to the originals
        for (&inputs, &origIn) |*val, *orig| {
            try std.testing.expectApproxEqAbs(orig.*, val.*, 1e-3);
        }
        for (&outputs, &origOut) |*val, *orig| {
            try std.testing.expectApproxEqAbs(orig.*, val.*, 1e-3);
        }
    }
};

// Run the tests for the norm structure
comptime {
    std.testing.refAllDecls(@This());
}
