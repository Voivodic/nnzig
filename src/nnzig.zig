//! Main module of the nnzig library. Provides the `NN` structure which serves as the
//! primary interface for creating, training, saving, and loading neural networks.
//! Training uses mini-batch gradient descent with the Adam optimizer.

// Import the modules used
const std = @import("std");
const act = @import("act");
const loss = @import("loss");
const eigen = @import("eigen");
const mlp = @import("mlp");
const params = @import("params");
const io = @import("io");
const norms = @import("norms");
const err = @import("errors").nnError;
const testing = std.testing;
const T = params.T;

/// Create the main structure for the NN
pub const NN = struct {
    allocator: std.mem.Allocator,
    ioContext: std.Io,
    nn: mlp.MLP,
    norm: norms.Norm,
    rand: std.Random,
    weights: []T = &.{},
    biases: []T = &.{},
    gradW: []T = &.{},
    gradB: []T = &.{},
    mW: []T = &.{},
    vW: []T = &.{},
    mB: []T = &.{},
    vB: []T = &.{},
    lossesTraining: []T = &.{},
    lossesValidation: []T = &.{},
    step: usize = 0,
    beta1_t: T = params.beta1,
    beta2_t: T = params.beta2,

    /// Initializes the neural network, allocating memory for weights, biases, gradients, and Adam optimizer moments
    pub fn init(allocator: std.mem.Allocator, ioContext: std.Io) !NN {
        // Initialize the random generator
        var xoshiro256 = std.Random.Xoshiro256.init(params.seed);

        // Create the NN
        var nn = NN{
            .allocator = allocator,
            .ioContext = ioContext,
            .nn = try mlp.MLP.init(
                allocator,
                &params.nNeurons,
            ),
            .norm = try norms.Norm.init(allocator),
            .rand = std.Random.Xoshiro256.random(&xoshiro256),
        };

        // Compute the total number of weights and biases
        var nWeights: usize = 0;
        var nBiases: usize = 0;
        for (1..params.nNeurons.len) |i| {
            nBiases += params.nNeurons[i];
            nWeights += params.nNeurons[i] * params.nNeurons[i - 1];
        }

        // Alloc memory for the weights
        if (allocator.alloc(T, nWeights)) |slice| {
            nn.weights = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the weights!\n", .{});
            return err.allocationOfWeights;
        }

        // Alloc memory for the gradients of the weights
        if (allocator.alloc(T, nWeights)) |slice| {
            nn.gradW = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the gradient of the weights!\n", .{});
            return err.allocationOfWeights;
        }

        // Alloc memory for the wt of the weights used in adam
        if (allocator.alloc(T, nWeights)) |slice| {
            nn.mW = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the mt of the weights!\n", .{});
            return err.allocationOfWeights;
        }

        // Alloc memory for the vt of the weights used in adam
        if (allocator.alloc(T, nWeights)) |slice| {
            nn.vW = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the vt of the weights!\n", .{});
            return err.allocationOfWeights;
        }

        // Set mW and vW to zero
        eigen.setZero(nn.mW);
        eigen.setZero(nn.vW);

        // Alloc memory for the biases
        if (allocator.alloc(T, nBiases)) |slice| {
            nn.biases = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the biases!\n", .{});
            return err.allocationOfBiases;
        }

        // Alloc memory for the gradients of the biases
        if (allocator.alloc(T, nBiases)) |slice| {
            nn.gradB = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the gradient of the biases!\n", .{});
            return err.allocationOfBiases;
        }

        // Alloc memory for the wt of the biases used in adam
        if (allocator.alloc(T, nBiases)) |slice| {
            nn.mB = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the mt of the biases!\n", .{});
            return err.allocationOfBiases;
        }

        // Alloc memory for the vt of the biases used in adam
        if (allocator.alloc(T, nBiases)) |slice| {
            nn.vB = slice;
        } else |_| {
            std.log.err("Failure when trying to allocate memory for the vt of the biases!\n", .{});
            return err.allocationOfBiases;
        }

        // Set mb and vb to zero
        eigen.setZero(nn.mB);
        eigen.setZero(nn.vB);

        // Set the parameters to normal random numbers
        for (nn.weights) |*weight| {
            weight.* = @floatCast(std.Random.floatNorm(nn.rand, f32));
        }
        for (nn.biases) |*bias| {
            bias.* = @floatCast(std.Random.floatNorm(nn.rand, f32));
        }

        // Create the slice for the losses of the training
        if (allocator.alloc(T, params.nEpochs)) |slice| {
            nn.lossesTraining = slice;
        } else |_| {
            std.log.err("Error while allocating the slice for the losses of the training!\n", .{});
            return err.lossesAllocation;
        }

        // Create the slice for the losses of the validation
        if (allocator.alloc(T, params.nEpochs)) |slice| {
            nn.lossesValidation = slice;
        } else |_| {
            std.log.err("Error while allocating the slice for the losses of the validation!\n", .{});
            return err.lossesAllocation;
        }

        // Set the losses to zero
        nn.zeroLosses();

        return nn;
    }

    /// Frees all memory allocated by `init`, including weights, biases, gradients, Adam moments, normalization data, and loss arrays
    pub fn deinit(self: *const NN) void {
        // Deinit the NN
        self.nn.deinit();

        // Deinit the slices used to optimize the weights by adam
        self.allocator.free(self.weights);
        self.allocator.free(self.gradW);
        self.allocator.free(self.mW);
        self.allocator.free(self.vW);

        // Deinit the slices used to optimize the biases by adam
        self.allocator.free(self.biases);
        self.allocator.free(self.gradB);
        self.allocator.free(self.mB);
        self.allocator.free(self.vB);

        // Free the memory used in the normalization
        self.norm.deinit();

        // Free the memory used in the losses array
        if (self.lossesTraining.len > 0) {
            self.allocator.free(self.lossesTraining);
            self.allocator.free(self.lossesValidation);
        }
    }

    test "[nnzig] init and deinit" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        const ioContext = std.testing.io;

        var nn = try NN.init(allocator, ioContext);
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

        for (nn.mW) |val| try testing.expectEqual(@as(T, 0.0), val);
        for (nn.vW) |val| try testing.expectEqual(@as(T, 0.0), val);
        for (nn.mB) |val| try testing.expectEqual(@as(T, 0.0), val);
        for (nn.vB) |val| try testing.expectEqual(@as(T, 0.0), val);

        try testing.expectEqual(@as(usize, params.nEpochs), nn.lossesTraining.len);
        try testing.expectEqual(@as(usize, params.nEpochs), nn.lossesValidation.len);
    }

    /// Computes Z-score normalization factors (mean and standard deviation) for the given inputs and outputs
    pub fn computeNormalization(self: *const NN, inputs: []const T, outputs: []const T) !void {
        try self.norm.computeNormalization(inputs, outputs);
    }

    test "[nnzig] computeNormalization" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        const ioContext = std.testing.io;
        const nn = try NN.init(allocator, ioContext);
        defer nn.deinit();

        const nIn: usize = params.nNeurons[0];
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];
        const nData: usize = 10;

        var inputs = try allocator.alloc(T, nData * nIn);
        var outputs = try allocator.alloc(T, nData * nOut);
        defer {
            allocator.free(inputs);
            allocator.free(outputs);
        }

        for (0..nData) |i| {
            inputs[i * nIn] = @floatFromInt(i + 1);
            inputs[i * nIn + 1] = @floatFromInt(2 * (i + 1));
            outputs[i * nOut] = @floatFromInt(3 * (i + 1));
            outputs[i * nOut + 1] = @floatFromInt(4 * (i + 1));
        }

        try nn.computeNormalization(inputs, outputs);

        try testing.expectEqual(@as(usize, nIn), nn.norm.aIn.len);
        try testing.expectEqual(@as(usize, nIn), nn.norm.bIn.len);
        try testing.expectEqual(@as(usize, nOut), nn.norm.aOut.len);
        try testing.expectEqual(@as(usize, nOut), nn.norm.bOut.len);

        for (nn.norm.aIn) |val| try testing.expect(val > 0);
        for (nn.norm.aOut) |val| try testing.expect(val > 0);
    }

    /// Normalizes the given inputs and outputs in-place using the previously computed normalization factors
    pub fn normalize(self: *const NN, inputs: []T, outputs: []T) !void {
        try self.norm.normalize(inputs, outputs);
    }

    test "[nnzig] normalize" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        const ioContext = std.testing.io;
        const nn = try NN.init(allocator, ioContext);
        defer nn.deinit();

        const nIn: usize = params.nNeurons[0];
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];
        const nData: usize = 10;

        var inputs = try allocator.alloc(T, nData * nIn);
        var outputs = try allocator.alloc(T, nData * nOut);
        defer {
            allocator.free(inputs);
            allocator.free(outputs);
        }

        for (0..nData) |i| {
            inputs[i * nIn] = @floatFromInt(i + 1);
            inputs[i * nIn + 1] = @floatFromInt(2 * (i + 1));
            outputs[i * nOut] = @floatFromInt(3 * (i + 1));
            outputs[i * nOut + 1] = @floatFromInt(4 * (i + 1));
        }

        try nn.computeNormalization(inputs, outputs);
        try nn.normalize(inputs, outputs);

        var sumIn: T = 0;
        for (0..nData) |i| {
            for (0..nIn) |j| {
                sumIn += inputs[i * nIn + j];
            }
        }
        const meanIn = sumIn / @as(T, @floatFromInt(nData * nIn));
        try testing.expect(@abs(meanIn) < 0.5);
    }

    /// Reverses the normalization, restoring data to its original scale
    pub fn denormalize(self: *const NN, inputs: []T, outputs: []T) !void {
        try self.norm.denormalize(inputs, outputs);
    }

    test "[nnzig] denormalize" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        const ioContext = std.testing.io;
        const nn = try NN.init(allocator, ioContext);
        defer nn.deinit();

        const nIn: usize = params.nNeurons[0];
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];
        const nData: usize = 10;

        var inputs = try allocator.alloc(T, nData * nIn);
        var outputs = try allocator.alloc(T, nData * nOut);
        defer {
            allocator.free(inputs);
            allocator.free(outputs);
        }

        for (0..nData) |i| {
            inputs[i * nIn] = @floatFromInt(i + 1);
            inputs[i * nIn + 1] = @floatFromInt(2 * (i + 1));
            outputs[i * nOut] = @floatFromInt(3 * (i + 1));
            outputs[i * nOut + 1] = @floatFromInt(4 * (i + 1));
        }

        try nn.computeNormalization(inputs, outputs);
        try nn.normalize(inputs, outputs);
        try nn.denormalize(inputs, outputs);

        const origIn = [_]T{ 1, 2, 2, 4, 3, 6, 4, 8, 5, 10, 6, 12, 7, 14, 8, 16, 9, 18, 10, 20 };
        const origOut = [_]T{ 3, 4, 6, 8, 9, 12, 12, 16, 15, 20, 18, 24, 21, 28, 24, 32, 27, 36, 30, 40 };

        const tol: T = switch (T) {
            f16 => 0.1,
            else => 1e-4,
        };
        for (0..nData * nIn) |i| {
            try testing.expectApproxEqAbs(origIn[i], inputs[i], tol);
        }
        for (0..nData * nOut) |i| {
            try testing.expectApproxEqAbs(origOut[i], outputs[i], tol);
        }
    }

    /// Resets all weight and bias gradients to zero
    pub fn zeroGrad(self: *const NN) void {
        eigen.setZero(self.gradW);
        eigen.setZero(self.gradB);
    }

    test "[nnzig] zeroGrad" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        const ioContext = std.testing.io;
        var nn = try NN.init(allocator, ioContext);
        defer nn.deinit();

        @memset(nn.gradW, 3.14);
        @memset(nn.gradB, 2.71);

        nn.zeroGrad();

        for (nn.gradW) |val| try testing.expectEqual(@as(T, 0.0), val);
        for (nn.gradB) |val| try testing.expectEqual(@as(T, 0.0), val);
    }

    /// Resets all training and validation loss arrays to zero
    pub fn zeroLosses(self: *const NN) void {
        eigen.setZero(self.lossesTraining);
        eigen.setZero(self.lossesValidation);
    }

    test "[nnzig] zeroLosses" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        const ioContext = std.testing.io;
        var nn = try NN.init(allocator, ioContext);
        defer nn.deinit();

        @memset(nn.lossesTraining, 5.0);
        @memset(nn.lossesValidation, 7.0);

        nn.zeroLosses();

        for (nn.lossesTraining) |val| try testing.expectEqual(@as(T, 0.0), val);
        for (nn.lossesValidation) |val| try testing.expectEqual(@as(T, 0.0), val);
    }

    /// Runs a forward pass through the network for the given input and returns the output slice
    pub fn forward(self: *const NN, input: []const T) ![]T {
        const nIn: usize = params.nNeurons[0];

        // Check the size of the input
        if (input.len % nIn != 0) {
            std.log.err("The input has a size non multiple of nNeurons[0]!\n", .{});
            return err.incompatibleSizes;
        }

        // Pass the input trough the NN
        return self.nn.forward(input, &params.nNeurons, self.weights, self.biases, &params.activations);
    }

    test "[nnzig] forward" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        const ioContext = std.testing.io;
        var nn = try NN.init(allocator, ioContext);
        defer nn.deinit();

        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];

        var input = [_]T{ 1.0, 2.0 };
        const output = try nn.forward(&input);

        try testing.expectEqual(nOut, output.len);
        for (output) |val| try testing.expect(std.math.isFinite(val));
    }

    test "[nnzig] gradient check vs finite differences (output layer)" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        const ioContext = std.testing.io;
        var nn = try NN.init(allocator, ioContext);
        defer nn.deinit();

        // Single fixed sample: backProp then yields the per-sample gradient
        // (no batch averaging to reason about).
        var inputs = [_]T{ 0.5, -0.3 };
        var outputs = [_]T{ 1.0, 0.0 };
        var dL = [_]T{ 0.0, 0.0 };

        _ = try nn.backProp(&inputs, &outputs);

        const eps: T = 1e-3;
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];
        const first_out_w: usize = nn.weights.len - nOut * params.nNeurons[params.nNeurons.len - 2];
        const first_out_b: usize = nn.biases.len - nOut;

        // Output layer uses the "none" activation, so perturbing its weights/
        // biases does not cross any ReLU kink -> finite differences are clean.
        const ow = nn.weights[first_out_w];
        const aw = nn.gradW[first_out_w];
        nn.weights[first_out_w] = ow + eps;
        const lp_w = loss.computeLoss(try nn.forward(&inputs), &outputs, &dL, params.lossFunc);
        nn.weights[first_out_w] = ow - eps;
        const lm_w = loss.computeLoss(try nn.forward(&inputs), &outputs, &dL, params.lossFunc);
        nn.weights[first_out_w] = ow;
        const fd_w: T = (lp_w - lm_w) / (2 * eps);

        const ob = nn.biases[first_out_b];
        const ab = nn.gradB[first_out_b];
        nn.biases[first_out_b] = ob + eps;
        const lp_b = loss.computeLoss(try nn.forward(&inputs), &outputs, &dL, params.lossFunc);
        nn.biases[first_out_b] = ob - eps;
        const lm_b = loss.computeLoss(try nn.forward(&inputs), &outputs, &dL, params.lossFunc);
        nn.biases[first_out_b] = ob;
        const fd_b: T = (lp_b - lm_b) / (2 * eps);

        std.log.info("gradCheck OUT: w analytic={} fd={} | b analytic={} fd={}", .{ aw, fd_w, ab, fd_b });

        try testing.expectApproxEqAbs(aw, fd_w, @abs(fd_w) * 1e-2 + 1e-4);
        try testing.expectApproxEqAbs(ab, fd_b, @abs(fd_b) * 1e-2 + 1e-4);
    }

    test "[nnzig] floatNorm init variance" {
        var xoshiro = std.Random.Xoshiro256.init(42);
        const rand = std.Random.Xoshiro256.random(&xoshiro);
        var sum: f64 = 0;
        var sumsq: f64 = 0;
        const n: usize = 200000;
        for (0..n) |_| {
            const v: f64 = @floatCast(std.Random.floatNorm(rand, f32));
            sum += v;
            sumsq += v * v;
        }
        const nf: f64 = @floatFromInt(n);
        const mean = sum / nf;
        const variance = sumsq / nf - mean * mean;
        std.log.info("floatNorm: mean={d:.4} variance={d:.4}", .{ mean, variance });
        try testing.expect(@abs(mean) < 0.05);
        try testing.expect(@abs(variance - 1.0) < 0.05);
    }

    /// Performs backpropagation over the given data and returns the average loss
    fn backProp(self: *const NN, inputs: []const T, outputs: []const T) !T {
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
            const pred: []T = try self.nn.forward(inputs[i * nIn .. (i + 1) * nIn], &params.nNeurons, self.weights, self.biases, &params.activations);

            // Compute the loss and its derivative
            lossTotal += loss.computeLoss(pred, outputs[i * nOut .. (i + 1) * nOut], dL, params.lossFunc) / nDataF;

            // Compute the gradient
            self.nn.backward(dL, &params.nNeurons, self.weights, self.biases, self.gradW, self.gradB);
        }

        // Normalize the gradients
        for (self.gradW) |*dw| {
            dw.* /= nDataF;
        }
        for (self.gradB) |*db| {
            db.* /= nDataF;
        }

        // Return the total loss computed
        return lossTotal;
    }

    /// Applies the bias-corrected Adam update rule to adjust weights and biases
    fn updateWeights(self: *NN) void {
        // Compute the normalization for mt and vt
        const normM: T = @floatCast(1.0 - self.beta1_t);
        const normV: T = @floatCast(1.0 - self.beta2_t);

        // Update the weights
        for (self.weights, self.gradW, self.mW, self.vW) |*w, *dw, *mw, *vw| {
            // Update the moments
            mw.* = params.beta1 * mw.* + (1.0 - params.beta1) * dw.*;
            vw.* = params.beta2 * vw.* + (1.0 - params.beta2) * (dw.*) * (dw.*);

            // Update the weights
            w.* -= ((mw.*) / normM) * params.lr / (@sqrt(vw.* / normV) + params.eps);
        }

        // Update the bias
        for (self.biases, self.gradB, self.mB, self.vB) |*b, *db, *mb, *vb| {
            // Update the moments
            mb.* = params.beta1 * mb.* + (1.0 - params.beta1) * db.*;
            vb.* = params.beta2 * vb.* + (1.0 - params.beta2) * (db.*) * (db.*);

            // Update the bias
            b.* -= ((mb.*) / normM) * params.lr / (@sqrt(vb.* / normV) + params.eps);
        }

        // Update the state of the Adam optimizer
        self.step += 1;
        self.beta1_t *= params.beta1;
        self.beta2_t *= params.beta2;
    }

    /// Trains the neural network using mini-batch gradient descent with the Adam optimizer. Splits data into training and validation sets, and records loss per epoch.
    pub fn train(self: *NN, inputs: []const T, outputs: []const T) !void {
        const nIn: usize = params.nNeurons[0];
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];

        // Check the dimensions of the inputs
        if (inputs.len % nIn != 0) {
            std.log.err("Incompatible dimension of inputs and number of neurons in layer 0!\n", .{});
            return err.incompatibleSizes;
        }

        // Check the dimensions of the outputs
        if (outputs.len % nOut != 0) {
            std.log.err("Incompatible dimension of outputs and number of neurons in layer -1!\n", .{});
            return err.incompatibleSizes;
        }

        // Check the number of inputs and outputs
        if (inputs.len / nIn != outputs.len / nOut) {
            std.log.err("Incompatible number of inputs and outputs!\n", .{});
            return err.incompatibleSizes;
        }

        // Zero the losses
        self.zeroLosses();

        // Alloc the dL array used in the computation of the loss of the validation
        var dL: []T = &.{};
        if (self.allocator.alloc(T, nOut)) |slice| {
            dL = slice;
        } else |_| {
            std.log.err("Error while allocating the slice for dL!", .{});
            return err.backProp;
        }
        defer self.allocator.free(dL);

        // Compute the dimension of the data
        const nData: usize = inputs.len / nIn;

        // Compute the size of training and validation
        const nTrain: usize = @as(usize, @intFromFloat(@as(T, @floatFromInt(nData)) * params.rTrain));
        const nValF: T = @as(T, @floatFromInt(nData)) * params.rVal;
        const nVal: usize = @as(usize, @intFromFloat(nValF));

        // Compute the number of batches
        const nBatches: usize = nTrain / params.batchSize;
        const nBatchesF: T = @as(T, @floatFromInt(nBatches));

        // Alloc the array of indexes used to shuffle the training data each epoch.
        // Only training indices [0, nTrain) are shuffled so that the validation set
        // stays fixed and comparable across epochs.
        var indexes: []usize = &.{};
        if (self.allocator.alloc(usize, nTrain)) |slice| {
            indexes = slice;
        } else |_| {
            std.log.err("Error while allocating the slice for the indexes!\n", .{});
            return err.backProp;
        }
        defer self.allocator.free(indexes);
        for (0..nTrain) |i| indexes[i] = i;

        // Alloc the batch buffers once and reuse them for every batch. Gathering the
        // shuffled samples into contiguous buffers keeps backProp cache-friendly.
        var inputsBatch: []T = &.{};
        if (self.allocator.alloc(T, params.batchSize * nIn)) |slice| {
            inputsBatch = slice;
        } else |_| {
            std.log.err("Error while allocating the slice for inputsBatch!\n", .{});
            return err.backProp;
        }
        defer self.allocator.free(inputsBatch);

        var outputsBatch: []T = &.{};
        if (self.allocator.alloc(T, params.batchSize * nOut)) |slice| {
            outputsBatch = slice;
        } else |_| {
            std.log.err("Error while allocating the slice for outputsBatch!\n", .{});
            return err.backProp;
        }
        defer self.allocator.free(outputsBatch);

        // Run over all epochs
        for (0..params.nEpochs) |epoch| {
            var lossEpoch: T = 0.0;

            // Shuffle the training data at the beginning of each epoch
            self.rand.shuffle(usize, indexes);

            // Run over the batches
            for (0..nBatches) |batch| {
                // Gather the shuffled samples into the contiguous batch buffers
                for (0..params.batchSize) |i| {
                    const idx = indexes[batch * params.batchSize + i];
                    @memcpy(inputsBatch[i * nIn .. (i + 1) * nIn], inputs[idx * nIn .. (idx + 1) * nIn]);
                    @memcpy(outputsBatch[i * nOut .. (i + 1) * nOut], outputs[idx * nOut .. (idx + 1) * nOut]);
                }

                // Compute the gradients
                if (self.backProp(inputsBatch, outputsBatch)) |lossE| {
                    lossEpoch += lossE / nBatchesF;
                } else |_| {
                    std.log.err("Problem when trying to backprop in epoch {} and batch {}!\n", .{ epoch, batch });
                    return err.backProp;
                }

                // Update the weights
                self.updateWeights();
            }

            // Handle the remainder batch
            const nRem: usize = nTrain % params.batchSize;
            if (nRem != 0) {
                // Gather the remaining shuffled samples into the batch buffers
                for (0..nRem) |i| {
                    const idx = indexes[nBatches * params.batchSize + i];
                    @memcpy(inputsBatch[i * nIn .. (i + 1) * nIn], inputs[idx * nIn .. (idx + 1) * nIn]);
                    @memcpy(outputsBatch[i * nOut .. (i + 1) * nOut], outputs[idx * nOut .. (idx + 1) * nOut]);
                }

                // Compute the gradients
                if (self.backProp(inputsBatch[0 .. nRem * nIn], outputsBatch[0 .. nRem * nOut])) |lossE| {
                    lossEpoch += lossE / nBatchesF;
                } else |_| {
                    std.log.err("Problem when trying to backprop in epoch {} and batch {}!\n", .{ epoch, nBatches });
                    return err.backProp;
                }

                // Update the weights
                self.updateWeights();
            }

            // Save the training loss of this epoch
            self.lossesTraining[epoch] = lossEpoch;

            // Save the loss of the validation of this epoch
            self.lossesValidation[epoch] = 0.0;
            for (0..nVal) |i| {
                const pred: []T = try self.nn.forward(inputs[(nTrain + i) * nIn .. (nTrain + i + 1) * nIn], &params.nNeurons, self.weights, self.biases, &params.activations);
                self.lossesValidation[epoch] += loss.computeLoss(pred, outputs[(nTrain + i) * nOut .. (nTrain + i + 1) * nOut], dL, params.lossFunc) / nValF;
            }

            // Print the current state
            if (params.printEvery > 0 and epoch % params.printEvery == 0) {
                std.log.info("Loss[{}] = ({}, {})\n", .{ epoch, self.lossesTraining[epoch], self.lossesValidation[epoch] });
            }
        }
    }

    test "[nnzig] train" {
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        const ioContext = std.testing.io;
        var nn = try NN.init(allocator, ioContext);
        defer nn.deinit();

        var xoshiro256 = std.Random.Xoshiro256.init(12345);
        const rand = std.Random.Xoshiro256.random(&xoshiro256);

        const nData: usize = 100;
        const nIn: usize = params.nNeurons[0];
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];

        var inputs = try allocator.alloc(T, nData * nIn);
        var outputs = try allocator.alloc(T, nData * nOut);
        defer {
            allocator.free(inputs);
            allocator.free(outputs);
        }

        for (0..nData) |i| {
            for (0..nIn) |j| {
                inputs[i * nIn + j] = @floatCast(std.Random.floatNorm(rand, f32));
            }
            const x0: f32 = @as(f32, inputs[i * nIn]);
            const x1: f32 = @as(f32, inputs[i * nIn + 1]);
            outputs[i * nOut] = @floatCast(2.0 + 1.2 * x0 - std.math.pow(f32, x1, 2) + @exp(-3.0 * x0 - 2.0 * x1));
            outputs[i * nOut + 1] = @floatCast(1.4 + 3.0 * x0 - std.math.pow(f32, x1, 2) + @exp(-2.0 * x0 - 3.0 * x1));
        }

        try nn.computeNormalization(inputs, outputs);
        try nn.normalize(inputs, outputs);
        try nn.train(inputs, outputs);

        try testing.expectEqual(@as(usize, params.nEpochs), nn.lossesTraining.len);
        try testing.expectEqual(@as(usize, params.nEpochs), nn.lossesValidation.len);
        try testing.expect(nn.lossesValidation[0] > nn.lossesValidation[params.nEpochs - 1]);
    }

    /// Save the weights of the neural network to a file
    pub fn saveWeights(self: *const NN, fileName: []const u8) !void {
        try io.saveWeights(fileName, self);
    }

    /// Load the weights of the neural network from a file
    pub fn loadWeights(self: *const NN, fileName: []const u8) !void {
        try io.loadWeights(fileName, self);
    }

    // Test the saving and loading of the weights
    test "[nnzig] save/load-Weights" {
        // Create the allocator
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        // Create the IO context
        const path = "test.bin";
        const ioContext = std.testing.io;

        // Create the neural network
        const nnIn = try NN.init(allocator, ioContext);
        defer nnIn.deinit();
        const nnOut = try NN.init(allocator, ioContext);
        defer nnOut.deinit();

        // Change some weights
        nnIn.weights[0] = 2.0;
        nnIn.biases[0] = 1.0;

        // Save the weights to a file
        try nnIn.saveWeights(path);

        // Load the weights from the same file
        try nnOut.loadWeights(path);

        // Check the weights
        try testing.expectEqual(nnIn.weights[0], nnOut.weights[0]);
        try testing.expectEqual(nnIn.biases[0], nnOut.biases[0]);

        // Delete the test file
        const cwd = std.Io.Dir.cwd();
        try cwd.deleteFile(ioContext, path);
    }

    /// Save data points to a binary file
    pub fn saveData(self: *const NN, fileName: []const u8, dataIn: []const T, dataOut: []const T) !void {
        try io.saveData(self.ioContext, fileName, dataIn, dataOut, params.nNeurons[0], params.nNeurons[params.nNeurons.len - 1]);
    }

    /// Load data points from a binary file
    // pub fn loadData(self: *const NN, fileName: []const u8) {
    //     try io.loadData(self.ioContext, fileName, self.dataIn, self.dataOut);
    // }

    /// Save the losses to a file
    pub fn saveLosses(self: *const NN, fileName: []const u8) !void {
        // Save the losses to a file
        try io.saveData(self.ioContext, fileName, self.lossesTraining, self.lossesValidation, 1, 1);
    }

    // Test the saving of the losses
    test "[nnzig] saveLosses" {
        // Create the allocator
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        // Create the IO context
        const path = "test.bin";
        const ioContext = std.testing.io;

        // Create the neural network
        const nn = try NN.init(allocator, ioContext);
        defer nn.deinit();

        // Set some values in the losses
        nn.lossesTraining[0] = 1.0;
        nn.lossesValidation[0] = 2.0;

        // Save the losses to a file
        try nn.saveLosses(path);

        // Load the losses from the same file
        var lossT: [params.nEpochs]T = undefined;
        var lossV: [params.nEpochs]T = undefined;
        try io.loadData(ioContext, path, &lossT, &lossV);

        // Check the losses
        try testing.expectEqual(nn.lossesTraining[0], lossT[0]);
        try testing.expectEqual(nn.lossesValidation[0], lossV[0]);

        // Delete the test file
        const cwd = std.Io.Dir.cwd();
        try cwd.deleteFile(ioContext, path);
    }
};

// Run the tests for the NN structure
comptime {
    std.testing.refAllDecls(@This());
}
