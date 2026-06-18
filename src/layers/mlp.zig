//! Multi-layer perceptron (MLP) implementation with forward and backward passes,
//! gradient computation, and Adam optimizer weight updates.
//! Uses Eigen-accelerated linear algebra for matrix-vector operations and
//! supports configurable activation functions per layer.

// Import the used modules
const std = @import("std");
const eigen = @import("eigen");
const act = @import("act");
const loss = @import("loss");
const params = @import("params");
const err = @import("errors").nnError;
const T = params.T;

/// Computes how much the variance is scaled by the activations.
/// Uses Kaiming/He variance scaling for ReLU layers, and Xavier/Glorot variance
/// scaling for tanh/sigmoid/linear layers.
fn computeSigma(a: params.activation, dimIn: usize, dimOut: usize) T {
    return switch (a) {
        .none => std.math.sqrt(2.0 / (@as(T, @floatFromInt(dimIn)) + @as(T, @floatFromInt(dimOut)))),
        .sigmoid => std.math.sqrt(2.0 / (@as(T, @floatFromInt(dimIn)) + @as(T, @floatFromInt(dimOut)))),
        .tanh => (5.0 / 3.0) * std.math.sqrt(2.0 / (@as(T, @floatFromInt(dimIn)) + @as(T, @floatFromInt(dimOut)))),
        .relu => std.math.sqrt(2.0 / @as(T, @floatFromInt(dimIn))),
    };
}


/// Define the structure for the MLP.
pub const MLP = struct {
    allocator: std.mem.Allocator,
    y: []T = &.{},
    dy: []T = &.{},
    V: []T = &.{},
    weights: []T = &.{},
    biases: []T = &.{},
    gradW: []T = &.{},
    gradB: []T = &.{},
    mW: []T = &.{},
    vW: []T = &.{},
    mB: []T = &.{},
    vB: []T = &.{},
    step: usize = 0,
    beta1_t: T = params.beta1,
    beta2_t: T = params.beta2,

    /// Initializes the MLP, allocating memory for all weights, biases, gradients,
    /// Adam optimizer moments, hidden layer activations, and scratch vectors.
    /// Weights and biases are initialized with random normal values.
    pub fn init(allocator: std.mem.Allocator) !MLP {
        // Create the mlp
        var mlp = MLP{
            .allocator = allocator,
        };

        // Compute the total number of hidden values
        var nHidden: usize = 0;
        for (params.nNeurons) |nN| {
            nHidden += nN;
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
        for (1..params.nNeurons.len) |i| {
            if (params.nNeurons[i] > vSize) {
                vSize = params.nNeurons[i];
            }
        }

        // Alloc memory for the V vector
        if (allocator.alloc(T, vSize)) |slice| {
            mlp.V = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for V!\n", .{});
            return err.allocationOfHiddens;
        }

        // Compute the total number of weights and biases
        var nWeights: usize = 0;
        var nBiases: usize = 0;
        for (1..params.nNeurons.len) |i| {
            nBiases += params.nNeurons[i];
            nWeights += params.nNeurons[i] * params.nNeurons[i - 1];
        }

        // Alloc memory for the weights
        if (allocator.alloc(T, nWeights)) |slice| {
            mlp.weights = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the weights!\n", .{});
            return err.allocationOfWeights;
        }

        // Alloc memory for the gradients of the weights
        if (allocator.alloc(T, nWeights)) |slice| {
            mlp.gradW = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the gradient of the weights!\n", .{});
            return err.allocationOfWeights;
        }

        // Alloc memory for the mt of the weights used in adam
        if (allocator.alloc(T, nWeights)) |slice| {
            mlp.mW = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the mt of the weights!\n", .{});
            return err.allocationOfWeights;
        }

        // Alloc memory for the vt of the weights used in adam
        if (allocator.alloc(T, nWeights)) |slice| {
            mlp.vW = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the vt of the weights!\n", .{});
            return err.allocationOfWeights;
        }

        // Set mW and vW to zero
        eigen.setZero(mlp.mW);
        eigen.setZero(mlp.vW);

        // Alloc memory for the biases
        if (allocator.alloc(T, nBiases)) |slice| {
            mlp.biases = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the biases!\n", .{});
            return err.allocationOfBiases;
        }

        // Alloc memory for the gradients of the biases
        if (allocator.alloc(T, nBiases)) |slice| {
            mlp.gradB = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the gradient of the biases!\n", .{});
            return err.allocationOfBiases;
        }

        // Alloc memory for the mt of the biases used in adam
        if (allocator.alloc(T, nBiases)) |slice| {
            mlp.mB = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the mt of the biases!\n", .{});
            return err.allocationOfBiases;
        }

        // Alloc memory for the vt of the biases used in adam
        if (allocator.alloc(T, nBiases)) |slice| {
            mlp.vB = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the vt of the biases!\n", .{});
            return err.allocationOfBiases;
        }

        // Set mB and vB to zero
        eigen.setZero(mlp.mB);
        eigen.setZero(mlp.vB);

        // Initialize each layer's weights with a variance-scaled normal distribution.
        var nprod: usize = 0;
        for (0..(params.nNeurons.len - 1)) |i| {
            const nin = params.nNeurons[i];
            const nout = params.nNeurons[i + 1];
            const seed = params.seed +% (i + 1);

            const sigma = computeSigma(params.activations[i], nin, nout);

            eigen.initWeights(mlp.weights[nprod .. nprod + nin * nout], sigma, seed);

            nprod += nin * nout;
        }

        // Initialize biases to zero. A non-zero random bias would add unit-variance
        eigen.setZero(mlp.biases);

        return mlp;
    }

    /// Frees all memory allocated by `init`.
    pub fn deinit(self: *const MLP) void {
        // Free V
        self.allocator.free(self.V);

        // Free the memory allocated for the hidden values and its derivatives
        self.allocator.free(self.y);
        self.allocator.free(self.dy);

        // Free the slices used to optimize the weights by adam
        self.allocator.free(self.weights);
        self.allocator.free(self.gradW);
        self.allocator.free(self.mW);
        self.allocator.free(self.vW);

        // Free the slices used to optimize the biases by adam
        self.allocator.free(self.biases);
        self.allocator.free(self.gradB);
        self.allocator.free(self.mB);
        self.allocator.free(self.vB);
    }

    test "[mlp] init and deinit" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        var mlp_inst = try MLP.init(allocator);
        defer mlp_inst.deinit();

        // Check hidden activation sizes
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

        // Check weight/bias/gradient sizes
        var nWeights: usize = 0;
        var nBiases: usize = 0;
        for (1..params.nNeurons.len) |i| {
            nWeights += params.nNeurons[i] * params.nNeurons[i - 1];
            nBiases += params.nNeurons[i];
        }

        try std.testing.expectEqual(nWeights, mlp_inst.weights.len);
        try std.testing.expectEqual(nWeights, mlp_inst.gradW.len);
        try std.testing.expectEqual(nWeights, mlp_inst.mW.len);
        try std.testing.expectEqual(nWeights, mlp_inst.vW.len);
        try std.testing.expectEqual(nBiases, mlp_inst.biases.len);
        try std.testing.expectEqual(nBiases, mlp_inst.gradB.len);
        try std.testing.expectEqual(nBiases, mlp_inst.mB.len);
        try std.testing.expectEqual(nBiases, mlp_inst.vB.len);

        // Check that Adam moments are zeroed
        for (mlp_inst.mW) |val| try std.testing.expectEqual(@as(T, 0.0), val);
        for (mlp_inst.vW) |val| try std.testing.expectEqual(@as(T, 0.0), val);
        for (mlp_inst.mB) |val| try std.testing.expectEqual(@as(T, 0.0), val);
        for (mlp_inst.vB) |val| try std.testing.expectEqual(@as(T, 0.0), val);

        // Check Adam state initialized
        try std.testing.expectEqual(@as(usize, 0), mlp_inst.step);
        try std.testing.expectEqual(params.beta1, mlp_inst.beta1_t);
        try std.testing.expectEqual(params.beta2, mlp_inst.beta2_t);
    }

    /// Performs a forward pass through all layers, returning the output slice.
    pub fn forward(self: *const MLP, input: []const T) ![]T {
        // Save the input in first hidden values
        eigen.setZero(self.dy);
        eigen.vectorInit(input, self.y);

        // Run over all layers and compute the partial result
        var nprod: usize = 0;
        var nsum: usize = 0;
        var nBias: usize = 0;
        for (0..(params.nNeurons.len - 1)) |i| {
            const nin = params.nNeurons[i];
            const nout = params.nNeurons[i + 1];

            // Compute the matrix vector multiplication then add the result to a third vector
            eigen.matrixVectorMulAdd(self.weights[nprod..(nprod + nin * nout)], self.y[nsum..(nsum + nin)], self.biases[nBias..(nBias + nout)], self.y[(nsum + nin)..(nsum + nin + nout)]);

            // Apply the activation function
            act.activateElements(self.y[(nsum + nin)..(nsum + nin + nout)], self.dy[(nsum + nin)..(nsum + nin + nout)], params.activations[i]);

            // Update the total sum and sum of products to keep track of the current position of the 1D slices
            nsum += nin;
            nprod += nin * nout;
            nBias += nout;
        }

        // Return the last slice of hidden values (the output)
        return self.y[nsum..];
    }

    test "[mlp] forward output size" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        var mlp_inst = try MLP.init(allocator);
        defer mlp_inst.deinit();

        const nIn = params.nNeurons[0];
        const nOut = params.nNeurons[params.nNeurons.len - 1];

        var input: [nIn]T = undefined;
        for (&input) |*val| val.* = 1.0;

        const output = try mlp_inst.forward(&input);

        try std.testing.expectEqual(nOut, output.len);
        for (output) |val| try std.testing.expect(std.math.isFinite(val));
    }

    /// Performs backpropagation from the loss gradient, computing weight and bias gradients for all layers.
    pub fn backward(self: *const MLP, dL: []const T) void {
        // Counters used to iterate over the weights and biases
        var layer: usize = params.nNeurons.len - 1;
        var nwIni: usize = self.weights.len - params.nNeurons[layer] * params.nNeurons[layer - 1];
        var nwEnd: usize = self.weights.len;
        var nbIni: usize = self.biases.len - params.nNeurons[layer];
        var nbEnd: usize = self.biases.len;
        var nyIni: usize = self.y.len - params.nNeurons[layer];
        var nyEnd: usize = self.y.len;

        // Compute the initial value for the vector V
        eigen.vectorInit(dL, self.V);

        // Iterate from the last to the first layer
        while (layer > 0) : (layer -= 1) {
            // Multiply the propagated vector V by the derivative of the activation function.
            eigen.vectorMul(self.dy[nyIni..nyEnd], self.V[0..(params.nNeurons[layer])]);

            // Update the gradient for the weights
            eigen.updateGradWeights(self.V[0..params.nNeurons[layer]], self.y[(nyIni - params.nNeurons[layer - 1])..nyIni], self.gradW[nwIni..nwEnd]);

            // Update the gradient for the biases
            eigen.updateGradBiases(self.V[0..params.nNeurons[layer]], self.gradB[nbIni..nbEnd]);

            // Do not run this part in the last iteration
            if (layer > 1) {

                // Update the M matrix using the current weight matrix
                eigen.vectorMatrixMul(self.V[0..(params.nNeurons[layer])], self.weights[nwIni..nwEnd]);

                // Update the counters
                nyEnd = nyIni;
                nyIni -= params.nNeurons[layer - 1];
                nbEnd = nbIni;
                nbIni -= params.nNeurons[layer - 1];
                nwEnd = nwIni;
                nwIni -= params.nNeurons[layer - 1] * params.nNeurons[layer - 2];
            }
        }
    }

    test "[mlp] backward produces non-zero gradients" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        var mlp_inst = try MLP.init(allocator);
        defer mlp_inst.deinit();

        const nIn = params.nNeurons[0];
        const nOut = params.nNeurons[params.nNeurons.len - 1];

        var input: [nIn]T = undefined;
        for (&input) |*val| val.* = 1.0;

        _ = try mlp_inst.forward(&input);

        var dL: [nOut]T = undefined;
        for (&dL) |*d| d.* = 1.0;

        mlp_inst.zeroGrad();
        mlp_inst.backward(&dL);

        var hasNonZeroGradW = false;
        for (mlp_inst.gradW) |val| {
            if (val != 0.0) {
                hasNonZeroGradW = true;
                break;
            }
        }
        try std.testing.expect(hasNonZeroGradW);

        var hasNonZeroGradB = false;
        for (mlp_inst.gradB) |val| {
            if (val != 0.0) {
                hasNonZeroGradB = true;
                break;
            }
        }
        try std.testing.expect(hasNonZeroGradB);
    }

    /// Resets all weight and bias gradients to zero.
    pub fn zeroGrad(self: *const MLP) void {
        eigen.setZero(self.gradW);
        eigen.setZero(self.gradB);
    }

    test "[mlp] zeroGrad" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        var mlp_inst = try MLP.init(allocator);
        defer mlp_inst.deinit();

        @memset(mlp_inst.gradW, 3.14);
        @memset(mlp_inst.gradB, 2.71);

        mlp_inst.zeroGrad();

        for (mlp_inst.gradW) |val| try std.testing.expectEqual(@as(T, 0.0), val);
        for (mlp_inst.gradB) |val| try std.testing.expectEqual(@as(T, 0.0), val);
    }

    /// Applies the bias-corrected Adam update rule to adjust weights and biases.
    pub fn updateWeights(self: *MLP) void {
        // Compute the normalization for mt and vt
        const normM: T = @floatCast(1.0 - self.beta1_t);
        const normV: T = @floatCast(1.0 - self.beta2_t);

        // Update the weights (moments + weight update done by the Eigen backend)
        eigen.adamUpdate(self.weights, self.gradW, self.mW, self.vW, params.beta1, params.beta2, params.lr, params.eps, normM, normV);

        // Update the biases (moments + bias update done by the Eigen backend)
        eigen.adamUpdate(self.biases, self.gradB, self.mB, self.vB, params.beta1, params.beta2, params.lr, params.eps, normM, normV);

        // Update the state of the Adam optimizer
        self.step += 1;
        self.beta1_t *= params.beta1;
        self.beta2_t *= params.beta2;
    }

    test "[mlp] updateWeights reduces loss" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        var mlp_inst = try MLP.init(allocator);
        defer mlp_inst.deinit();

        const nIn: usize = params.nNeurons[0];
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];

        var inputs: [nIn]T = .{ 0.5, -0.3 };
        var outputs: [nOut]T = .{ 1.0, 0.0 };
        var dL: [nOut]T = .{ 0.0, 0.0 };

        // Compute initial loss
        const loss0 = loss.computeLoss(try mlp_inst.forward(&inputs), &outputs, &dL, params.lossFunc);

        // Compute gradients and update weights a few times
        for (0..50) |_| {
            _ = try mlp_inst.updateGrads(&inputs, &outputs);
            mlp_inst.updateWeights();
        }

        // Compute final loss
        const loss1 = loss.computeLoss(try mlp_inst.forward(&inputs), &outputs, &dL, params.lossFunc);

        try std.testing.expect(loss1 < loss0);
    }

    /// Computes gradients over the given batch of data and returns the average loss.
    pub fn updateGrads(self: *MLP, inputs: []const T, outputs: []const T) !T {
        // Get the data size
        const nIn: usize = params.nNeurons[0];
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];
        const nDataF: T = @as(T, @floatFromInt(inputs.len / nIn));
        const nData: usize = @as(usize, @intFromFloat(nDataF));

        self.zeroGrad();

        // Slice to keep the derivatives of the loss
        var dL: []T = undefined;
        if (self.allocator.alloc(T, nOut)) |slice| {
            dL = slice;
        } else |_| {
            std.log.err("Problem to allocate the array for the derivatives of the loss!\n", .{});
            return err.backProp;
        }
        defer self.allocator.free(dL);

        // Save the value of the total loss
        var lossTotal: T = 0.0;

        // Run over all inputs and outputs
        for (0..nData) |i| {
            // Compute the forward pass
            const pred: []T = try self.forward(inputs[i * nIn .. (i + 1) * nIn]);

            // Compute the loss and its derivative
            lossTotal += loss.computeLoss(pred, outputs[i * nOut .. (i + 1) * nOut], dL, params.lossFunc) / nDataF;

            // Compute the gradient
            self.backward(dL);
        }

        // Normalize the gradients (done by the Eigen backend)
        eigen.divScalar(self.gradW, nDataF);
        eigen.divScalar(self.gradB, nDataF);

        // Return the total loss computed
        return lossTotal;
    }
};

// Run the tests for the MLP structure
comptime {
    std.testing.refAllDecls(@This());
}
