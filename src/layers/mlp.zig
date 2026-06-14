//! Multi-layer perceptron (MLP) implementation with forward and backward passes.
//! Uses Eigen-accelerated linear algebra for matrix-vector operations and
//! supports configurable activation functions per layer.

// Import the used modules
const std = @import("std");
const eigen = @import("eigen");
const act = @import("act");
const params = @import("params");
const err = @import("errors").nnError;
const T = params.T;

/// Define the structure for the MLP
pub const MLP = struct {
    allocator: std.mem.Allocator,
    y: []T = &.{},
    dy: []T = &.{},
    V: []T = &.{},

    /// Initializes the MLP, allocating memory for hidden layer activations, their derivatives, and a scratch vector
    pub fn init(allocator: std.mem.Allocator, nNeurons: []const usize) !MLP {
        // Create the mlp
        var mlp = MLP{
            .allocator = allocator,
        };

        // Compute the total number of hidden values
        var nHidden: usize = 0;
        for (nNeurons) |*nN| {
            nHidden += nN.*;
        }

        // Alloc space for the hidden values
        if (allocator.alloc(T, nHidden)) |slice| {
            mlp.y = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for y!\n", .{});
            return err.allocationOfHiddens;
        }

        // Alloc space for the derivative of hidden values
        if (allocator.alloc(T, nHidden)) |slice| {
            mlp.dy = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for dy!\n", .{});
            return err.allocationOfHiddens;
        }

        // Compute the size of the largest V vector used by adam
        var vSize: usize = 0;
        for (1..nNeurons.len) |i| {
            if (nNeurons[i] > vSize) {
                vSize = nNeurons[i];
            }
        }

        // Alloc memory for the V vector
        if (allocator.alloc(T, vSize)) |slice| {
            mlp.V = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for V!\n", .{});
            return err.allocationOfHiddens;
        }

        return mlp;
    }

    /// Frees all memory allocated by `init`
    pub fn deinit(self: *const MLP) void {
        // Free V
        self.allocator.free(self.V);

        // Free the memory allocated for the hidden values and its derivatives
        self.allocator.free(self.y);
        self.allocator.free(self.dy);
    }

    /// Performs a forward pass through all layers, returning the output slice
    pub fn forward(self: *const MLP, input: []const T, nNeurons: []const usize, weights: []const T, biases: []const T, activations: []const params.activation) ![]T {
        // Save the input in first hidden values
        for (input, 0..input.len) |*in, i| {
            self.y[i] = in.*;
            self.dy[i] = 0.0;
        }

        // Run over all layers and compute the partial retult
        var nprod: usize = 0;
        var nsum: usize = 0;
        var nBias: usize = 0;
        for (0..(nNeurons.len - 1)) |i| {
            const nin = nNeurons[i];
            const nout = nNeurons[i + 1];

            // Compute the matrix vector multiplication then add the result to a third vector
            eigen.matrixVectorMulAdd(weights[nprod..(nprod + nin * nout)], self.y[nsum..(nsum + nin)], biases[nBias..(nBias + nout)], self.y[(nsum + nin)..(nsum + nin + nout)]);

            // Apply the activation function
            try act.activateElements(self.y[(nsum + nin)..(nsum + nin + nout)], self.dy[(nsum + nin)..(nsum + nin + nout)], activations[i]);

            // Update the total sum and sum of products to keep track of the current position of the 1D slices
            nsum += nin;
            nprod += nin * nout;
            nBias += nout;
        }

        // Return the last slice of hidden values (the output)
        return self.y[nsum..];
    }

    /// Performs backpropagation from the loss gradient, computing weight and bias gradients for all layers
    pub fn backward(self: *const MLP, dL: []const T, nNeurons: []const usize, weights: []const T, biases: []const T, gradW: []T, gradB: []T) void {
        // Counters used to iterate over the weights and biases
        var layer: usize = nNeurons.len - 1;
        var nwIni: usize = weights.len - nNeurons[layer] * nNeurons[layer - 1];
        var nwEnd: usize = weights.len;
        var nbIni: usize = biases.len - nNeurons[layer];
        var nbEnd: usize = biases.len;
        var nyIni: usize = self.y.len - nNeurons[layer];
        var nyEnd: usize = self.y.len;

        // Compute the initial value for the vector V
        eigen.vectorInit(dL, self.V);

        // Iterate from the last to the first layer
        while (layer > 0) : (layer -= 1) {
            // Multiply the propagated vector V by the derivative of the activation function
            eigen.vectorMul(self.V[0..(nNeurons[layer])], self.dy[nyIni..nyEnd]);

            // Update the gradient for the weights
            eigen.updateGradWeights(self.V[0..nNeurons[layer]], self.y[(nyIni - nNeurons[layer - 1])..nyIni], gradW[nwIni..nwEnd]);

            // Update the gradient for the biases
            eigen.updateGradBiases(self.V[0..nNeurons[layer]], gradB[nbIni..nbEnd]);

            // Do not run this part in the last iteration
            if (layer > 1) {

                // Update the M matrix using the current weight matrix
                eigen.vectorMatrixMul(self.V[0..(nNeurons[layer])], weights[nwIni..nwEnd]);

                // Update the counters
                nyEnd = nyIni;
                nyIni -= nNeurons[layer - 1];
                nbEnd = nbIni;
                nbIni -= nNeurons[layer - 1];
                nwEnd = nwIni;
                nwIni -= nNeurons[layer - 1] * nNeurons[layer - 2];
            }
        }
    }

    test "[mlp] init and deinit" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        var mlp_inst = try MLP.init(allocator, &params.nNeurons);
        defer mlp_inst.deinit();

        var nHidden: usize = 0;
        for (params.nNeurons) |n| nHidden += n;

        try std.testing.expectEqual(nHidden, mlp_inst.y.len);
        try std.testing.expectEqual(nHidden, mlp_inst.dy.len);

        var vSize: usize = 0;
        for (1..params.nNeurons.len) |i| {
            if (params.nNeurons[i] > vSize) {
                vSize = params.nNeurons[i];
            }
        }
        try std.testing.expectEqual(vSize, mlp_inst.V.len);
    }

    test "[mlp] forward output size" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        var mlp_inst = try MLP.init(allocator, &params.nNeurons);
        defer mlp_inst.deinit();

        const nIn = params.nNeurons[0];
        const nOut = params.nNeurons[params.nNeurons.len - 1];

        comptime var nWeights: usize = 0;
        comptime var nBiases: usize = 0;
        inline for (1..params.nNeurons.len) |i| {
            nWeights += params.nNeurons[i] * params.nNeurons[i - 1];
            nBiases += params.nNeurons[i];
        }

        var weights: [nWeights]T = undefined;
        var biases: [nBiases]T = undefined;
        for (&weights) |*w| w.* = 0.01;
        for (&biases) |*b| b.* = 0.0;

        var input: [nIn]T = undefined;
        for (&input) |*val| val.* = 1.0;

        const output = try mlp_inst.forward(&input, &params.nNeurons, &weights, &biases, &params.activations);

        try std.testing.expectEqual(nOut, output.len);
        for (output) |val| try std.testing.expect(std.math.isFinite(val));
    }

    test "[mlp] backward produces non-zero gradients" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        var mlp_inst = try MLP.init(allocator, &params.nNeurons);
        defer mlp_inst.deinit();

        const nIn = params.nNeurons[0];
        const nOut = params.nNeurons[params.nNeurons.len - 1];

        comptime var nWeights: usize = 0;
        comptime var nBiases: usize = 0;
        inline for (1..params.nNeurons.len) |i| {
            nWeights += params.nNeurons[i] * params.nNeurons[i - 1];
            nBiases += params.nNeurons[i];
        }

        var weights: [nWeights]T = undefined;
        var biases: [nBiases]T = undefined;
        var gradW: [nWeights]T = undefined;
        var gradB: [nBiases]T = undefined;

        var xoshiro256 = std.Random.Xoshiro256.init(42);
        const rand = std.Random.Xoshiro256.random(&xoshiro256);
        for (&weights) |*w| w.* = @floatCast(std.Random.floatNorm(rand, f32));
        for (&biases) |*b| b.* = 0.0;
        for (&gradW) |*g| g.* = 0.0;
        for (&gradB) |*g| g.* = 0.0;

        var input: [nIn]T = undefined;
        for (&input) |*val| val.* = 1.0;

        _ = try mlp_inst.forward(&input, &params.nNeurons, &weights, &biases, &params.activations);

        var dL: [nOut]T = undefined;
        for (&dL) |*d| d.* = 1.0;

        mlp_inst.backward(&dL, &params.nNeurons, &weights, &biases, &gradW, &gradB);

        var hasNonZeroGradW = false;
        for (gradW) |val| {
            if (val != 0.0) {
                hasNonZeroGradW = true;
                break;
            }
        }
        try std.testing.expect(hasNonZeroGradW);

        var hasNonZeroGradB = false;
        for (gradB) |val| {
            if (val != 0.0) {
                hasNonZeroGradB = true;
                break;
            }
        }
        try std.testing.expect(hasNonZeroGradB);
    }
};

// Run the tests for the NN structure
comptime {
    std.testing.refAllDecls(@This());
}

