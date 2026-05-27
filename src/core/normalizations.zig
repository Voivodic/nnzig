//! This module is used to normalize and denormalize the data
//! The data is normalized by x' = (x - b)/a => x = x'*a + b

// Import the moduels used
const std = @import("std");
const params = @import("params");
const err = @import("errors").normalizationError;

// Computes the slope and shift for the normalization using a Z-score standardization
// a = std and b = mean
fn computeMeanStd(comptime T: type, norm: *const Norm(T), inputs: [][]T, outputs: [][]T) void {
    // Get the number of data points
    const N: T = @as(T, @floatFromInt(inputs.len));

    // Compute means
    for (inputs, outputs) |*ins, *outs| {
        for (ins.*, norm.bIn) |*in, *mu| {
            mu.* += in.* / N;
        }
        for (outs.*, norm.bOut) |*out, *mu| {
            mu.* += out.* / N;
        }
    }

    // Compute the stds
    for (inputs, outputs) |*ins, *outs| {
        for (ins.*, norm.bIn, norm.aIn) |*in, *mu, *st| {
            st.* += (in.* - mu.*) * (in.* - mu.*) / N;
        }
        for (outs.*, norm.bOut, norm.aOut) |*out, *mu, *st| {
            st.* += (out.* - mu.*) * (out.* - mu.*) / N;
        }
    }
}

// Function used by each thread to normalize the slices
fn normalizeChunk(comptime T: type, norm: *const Norm(T), inputs: [][]T, outputs: [][]T) void {
    // Normalize the inputs and outputs
    for (inputs, outputs) |*ins, *outs| {
        for (ins.*, norm.aIn, norm.bIn) |*in, *a, *b| {
            in.* = (in.* - b.*) / a.*;
        }
        for (outs.*, norm.aOut, norm.bOut) |*out, *a, *b| {
            out.* = (out.* - b.*) / a.*;
        }
    }
}

// Function used by each thread to denormalize the slices
fn denormalizeChunk(comptime T: type, norm: *const Norm(T), inputs: [][]T, outputs: [][]T) void {
    // Denormalize the inputs and outputs
    for (inputs, outputs) |*ins, *outs| {
        for (ins.*, norm.aIn, norm.bIn) |*in, *a, *b| {
            in.* = (in.*) * (a.*) + b.*;
        }
        for (outs.*, norm.aOut, norm.bOut) |*out, *a, *b| {
            out.* = (out.*) * a.* + b.*;
        }
    }
}

/// Main structure to handle the normalization and denormalization
pub fn Norm(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        aIn: []T = &.{},
        bIn: []T = &.{},
        aOut: []T = &.{},
        bOut: []T = &.{},

        /// Initializes the structure computing the mean and std of the inputs and outputs
        pub fn init(allocator: std.mem.Allocator) !Norm(T) {
            // Create the structure to be outputed
            var norm = Norm(T){
                .allocator = allocator,
            };

            // Get the size of the inputs and outputs
            const nInputs = params.nNeurons[0];
            const nOutputs = params.nNeurons[params.nNeurons.len - 1];

            // Alloc the slices for the mean and std of inputs and outputs
            if (norm.allocator.alloc(T, nInputs)) |slice| {
                norm.aIn = slice;
            } else |_| {
                std.debug.print("Problem in the allocation of the mean of the inputs!\n", .{});
                return err.aAllocation;
            }

            // Allocate the slice for the std of the input
            if (norm.allocator.alloc(T, nInputs)) |slice| {
                norm.bIn = slice;
            } else |_| {
                std.debug.print("Problem in the allocation of the std of the inputs!\n", .{});
                return err.bAllocation;
            }

            // Zero the input arrays
            for (norm.aIn, norm.bIn) |*a, *b| {
                a.* = 0.0;
                b.* = 0.0;
            }

            // Allocate the slice for the mean of the outputs
            if (norm.allocator.alloc(T, nOutputs)) |slice| {
                norm.aOut = slice;
            } else |_| {
                std.debug.print("Problem in the allocation of the mean of the outputs!\n", .{});
                return err.aAllocation;
            }

            // Allocate the slice for the std of the outputs
            if (norm.allocator.alloc(T, nOutputs)) |slice| {
                norm.bOut = slice;
            } else |_| {
                std.debug.print("Problem in the allocation of the std of the outputs!\n", .{});
                return err.bAllocation;
            }

            // Zero the outputs arrys
            for (norm.aOut, norm.bOut) |*a, *b| {
                a.* = 0.0;
                b.* = 0.0;
            }

            return norm;
        }

        /// Frees all memory allocated in the structure
        pub fn deinit(self: *const Norm(T)) void {
            self.allocator.free(self.aIn);
            self.allocator.free(self.bIn);
            self.allocator.free(self.aOut);
            self.allocator.free(self.bOut);
        }

        /// Computes the normalization factors
        pub fn computeNormalization(self: *const Norm(T), inputs: [][]T, outputs: [][]T) !void {
            // Check if the size of inputs and outputs are the same
            if (inputs.len != outputs.len) {
                std.debug.print("Inputs and outputs must have the same number of data vectors!\n", .{});
                return err.incompatibleSizes;
            }

            // Check if the normalizations were computed
            if (self.aIn.len == 0) {
                std.debug.print("The normalization structure was not initialized!\n", .{});
                return err.notInitialized;
            }

            // Compute the slope and shift for the inputs and outputs
            computeMeanStd(T, self, inputs, outputs);
            // switch (params.normType) {
            //     .meanStd => computeMeanStd(T, self, inputs, outputs),
            //     else => {
            //         std.debug.print("Normalization type not implemented!\n", .{});
            //         return err.notImplemented;
            //     },
            // }
        }

        /// Normalizes the inputs and outputs given
        pub fn normalize(self: *const Norm(T), inputs: [][]T, outputs: [][]T) !void {
            // Check if the size of inputs and outputs are the same
            if (inputs.len != outputs.len) {
                std.debug.print("Inputs and outputs must have the same number of data vectors!\n", .{});
                return err.incompatibleSizes;
            }

            // Check if the normalizations were computed
            if (self.aIn.len == 0) {
                std.debug.print("The normalization structure was not initialized!\n", .{});
                return err.notInitialized;
            }

            // Compute the chunck size
            const chunkSize: usize = (inputs.len + params.numThreads - 1) / params.numThreads;

            // Define the array of threads
            var threads: [params.numThreads]std.Thread = undefined;

            // Compute the chuncks for each thread
            for (&threads, 0..params.numThreads) |*thread, i| {
                const start: usize = i * chunkSize;
                const end: usize = @min(start + chunkSize, inputs.len);

                // Normalize the inputs and outputs in the current chunk
                if (std.Thread.spawn(.{}, normalizeChunk, .{ T, self, inputs[start..end], outputs[start..end] })) |t| {
                    thread.* = t;
                } else |_| {
                    std.debug.print("Error in spawning thread {}!\n", .{i});
                    return err.threadRun;
                }
            }

            // Await all threads to finish
            for (&threads) |*thread| {
                thread.*.join();
            }
        }

        /// Denormalize the inputs and outputs
        pub fn denormalize(self: *const Norm(T), inputs: [][]T, outputs: [][]T) !void {
            // Check if the size of inputs and outputs are the same
            if (inputs.len != outputs.len) {
                std.debug.print("Inputs and outputs must have the same number of data vectors!\n", .{});
                return err.IncompatibleSizes;
            }

            // Check if the normalizations were computed
            if (self.bIn.len == 0) {
                std.debug.print("The normalization structure was not initialized!\n", .{});
                return err.notInitialized;
            }

            // Compute the chunck size
            const chunkSize: usize = (inputs.len + params.numThreads - 1) / params.numThreads;

            // Define the array of threads
            var threads: [params.numThreads]std.Thread = undefined;

            // Compute the chuncks for each thread
            for (&threads, 0..params.numThreads) |*thread, i| {
                const start: usize = i * chunkSize;
                const end: usize = @min(start + chunkSize, inputs.len);

                // Denormalize the inputs and outputs in the current chunk
                if (std.Thread.spawn(.{}, denormalizeChunk, .{ T, self, inputs[start..end], outputs[start..end] })) |t| {
                    thread.* = t;
                } else |_| {
                    std.debug.print("Error in spawning thread {}!\n", .{i});
                    return err.threadRun;
                }
            }

            // Await all threads to finish
            for (&threads) |*thread| {
                thread.*.join();
            }
        }
    };
}
