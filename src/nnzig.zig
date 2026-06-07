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
const fType = params.floatType;

/// Create the main structure for the NN
pub const NN = struct {
    allocator: std.mem.Allocator,
    ioContext: std.Io,
    nn: mlp.MLP,
    norm: norms.Norm,
    rand: std.Random,
    weights: []fType = &.{},
    biases: []fType = &.{},
    gradW: []fType = &.{},
    gradB: []fType = &.{},
    mW: []fType = &.{},
    vW: []fType = &.{},
    mB: []fType = &.{},
    vB: []fType = &.{},
    lossesTraining: []fType = &.{},
    lossesValidation: []fType = &.{},

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
        if (allocator.alloc(fType, nWeights)) |slice| {
            nn.weights = slice;
        } else |_| {
            std.debug.print("Failure when trying to allocate memory for the weights!\n", .{});
            return err.allocationOfWeights;
        }

        // Alloc memory for the gradients of the weights
        if (allocator.alloc(fType, nWeights)) |slice| {
            nn.gradW = slice;
        } else |_| {
            std.debug.print("Failure when trying to allocate memory for the gradient of the weights!\n", .{});
            return err.allocationOfWeights;
        }

        // Alloc memory for the wt of the weights used in adam
        if (allocator.alloc(fType, nWeights)) |slice| {
            nn.mW = slice;
        } else |_| {
            std.debug.print("Failure when trying to allocate memory for the mt of the weights!\n", .{});
            return err.allocationOfWeights;
        }

        // Alloc memory for the vt of the weights used in adam
        if (allocator.alloc(fType, nWeights)) |slice| {
            nn.vW = slice;
        } else |_| {
            std.debug.print("Failure when trying to allocate memory for the vt of the weights!\n", .{});
            return err.allocationOfWeights;
        }

        // Set mW and vW to zero
        eigen.setZero(fType, nn.mW);
        eigen.setZero(fType, nn.vW);

        // Alloc memory for the biases
        if (allocator.alloc(fType, nBiases)) |slice| {
            nn.biases = slice;
        } else |_| {
            std.debug.print("Failure when trying to allocate memory for the biases!\n", .{});
            return err.allocationOfBiases;
        }

        // Alloc memory for the gradients of the biases
        if (allocator.alloc(fType, nBiases)) |slice| {
            nn.gradB = slice;
        } else |_| {
            std.debug.print("Failure when trying to allocate memory for the gradient of the biases!\n", .{});
            return err.allocationOfBiases;
        }

        // Alloc memory for the wt of the biases used in adam
        if (allocator.alloc(fType, nBiases)) |slice| {
            nn.mB = slice;
        } else |_| {
            std.debug.print("Failure when trying to allocate memory for the mt of the biases!\n", .{});
            return err.allocationOfBiases;
        }

        // Alloc memory for the vt of the biases used in adam
        if (allocator.alloc(fType, nBiases)) |slice| {
            nn.vB = slice;
        } else |_| {
            std.debug.print("Failure when trying to allocate memory for the vt of the biases!\n", .{});
            return err.allocationOfBiases;
        }

        // Set mb and vb to zero
        eigen.setZero(fType, nn.mB);
        eigen.setZero(fType, nn.vB);

        // Set the parameters to normal random numbers
        for (nn.weights) |*weight| {
            weight.* = std.Random.floatNorm(nn.rand, fType);
        }
        for (nn.biases) |*bias| {
            bias.* = std.Random.floatNorm(nn.rand, fType);
        }

        // Create the slice for the losses of the training
        if (allocator.alloc(fType, params.nEpochs)) |slice| {
            nn.lossesTraining = slice;
        } else |_| {
            std.debug.print("Error while allocating the slice for the losses of the training!\n", .{});
            return err.lossesAllocation;
        }

        // Create the slice for the losses of the validation
        if (allocator.alloc(fType, params.nEpochs)) |slice| {
            nn.lossesValidation = slice;
        } else |_| {
            std.debug.print("Error while allocating the slice for the losses of the validation!\n", .{});
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

        for (nn.mW) |val| try testing.expectEqual(@as(fType, 0.0), val);
        for (nn.vW) |val| try testing.expectEqual(@as(fType, 0.0), val);
        for (nn.mB) |val| try testing.expectEqual(@as(fType, 0.0), val);
        for (nn.vB) |val| try testing.expectEqual(@as(fType, 0.0), val);

        try testing.expectEqual(@as(usize, params.nEpochs), nn.lossesTraining.len);
        try testing.expectEqual(@as(usize, params.nEpochs), nn.lossesValidation.len);
    }

    /// Computes Z-score normalization factors (mean and standard deviation) for the given inputs and outputs
    pub fn computeNormalization(self: *const NN, inputs: []const fType, outputs: []const fType) !void {
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

        var inputs = try allocator.alloc(fType, nData * nIn);
        var outputs = try allocator.alloc(fType, nData * nOut);
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
    pub fn normalize(self: *const NN, inputs: []fType, outputs: []fType) !void {
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

        var inputs = try allocator.alloc(fType, nData * nIn);
        var outputs = try allocator.alloc(fType, nData * nOut);
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

        var sumIn: fType = 0;
        for (0..nData) |i| {
            for (0..nIn) |j| {
                sumIn += inputs[i * nIn + j];
            }
        }
        const meanIn = sumIn / @as(fType, @floatFromInt(nData * nIn));
        try testing.expect(@abs(meanIn) < 0.5);
    }

    /// Reverses the normalization, restoring data to its original scale
    pub fn denormalize(self: *const NN, inputs: []fType, outputs: []fType) !void {
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

        var inputs = try allocator.alloc(fType, nData * nIn);
        var outputs = try allocator.alloc(fType, nData * nOut);
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

        const origIn = [_]fType{ 1, 2, 2, 4, 3, 6, 4, 8, 5, 10, 6, 12, 7, 14, 8, 16, 9, 18, 10, 20 };
        const origOut = [_]fType{ 3, 4, 6, 8, 9, 12, 12, 16, 15, 20, 18, 24, 21, 28, 24, 32, 27, 36, 30, 40 };

        const tol: fType = 1e-4;
        for (0..nData * nIn) |i| {
            try testing.expectApproxEqAbs(origIn[i], inputs[i], tol);
        }
        for (0..nData * nOut) |i| {
            try testing.expectApproxEqAbs(origOut[i], outputs[i], tol);
        }
    }

    /// Resets all weight and bias gradients to zero
    pub fn zeroGrad(self: *const NN) void {
        eigen.setZero(fType, self.gradW);
        eigen.setZero(fType, self.gradB);
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

        for (nn.gradW) |val| try testing.expectEqual(@as(fType, 0.0), val);
        for (nn.gradB) |val| try testing.expectEqual(@as(fType, 0.0), val);
    }

    /// Resets all training and validation loss arrays to zero
    pub fn zeroLosses(self: *const NN) void {
        eigen.setZero(fType, self.lossesTraining);
        eigen.setZero(fType, self.lossesValidation);
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

        for (nn.lossesTraining) |val| try testing.expectEqual(@as(fType, 0.0), val);
        for (nn.lossesValidation) |val| try testing.expectEqual(@as(fType, 0.0), val);
    }

    /// Runs a forward pass through the network for the given input and returns the output slice
    pub fn forward(self: *const NN, input: []const fType) ![]fType {
        const nIn: usize = params.nNeurons[0];

        // Check the size of the input
        if (input.len % nIn != 0) {
            std.debug.print("The input has a size non multiple of nNeurons[0]!\n", .{});
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

        var input = [_]fType{ 1.0, 2.0 };
        const output = try nn.forward(&input);

        try testing.expectEqual(nOut, output.len);
        for (output) |val| try testing.expect(std.math.isFinite(val));
    }

    /// Performs backpropagation over the given data and returns the average loss
    fn backProp(self: *const NN, inputs: []const fType, outputs: []const fType) !fType {
        // Get the data size
        const nIn: usize = params.nNeurons[0];
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];
        const nDataF: fType = @as(fType, @floatFromInt(inputs.len / nIn));
        const nData: usize = @as(usize, @intFromFloat(nDataF));

        self.zeroGrad();

        // Slice to keep the derivatives of the loss
        var dL: []fType = undefined;
        if (self.allocator.alloc(fType, nOut)) |slice| {
            dL = slice;
        } else |_| {
            std.debug.print("Problem to allocate the array for the derivatives of the loss!\n", .{});
            return err.backProp;
        }
        defer self.allocator.free(dL);

        // Save the value of the total loss
        var lossTotal: fType = 0.0;

        // Run over all inputs and outputs
        for (0..nData) |i| {
            // Compute the forward pass
            const pred: []fType = try self.nn.forward(inputs[i * nIn .. (i + 1) * nIn], &params.nNeurons, self.weights, self.biases, &params.activations);

            // Compute the loss and its derivative
            lossTotal += try loss.computeLoss(pred, outputs[i * nOut .. (i + 1) * nOut], dL, params.lossFunc) / nDataF;

            // Compute the gradient
            self.nn.backward(dL, &params.nNeurons, self.weights, self.biases, self.gradW, self.gradB);
        }

        // Return the total loss computed
        return lossTotal;
    }

    /// Updates the first and second moment estimates (m and v) for the Adam optimizer
    fn updateMV(self: *const NN) void {
        // Update the weights
        for (self.gradW, self.mW, self.vW) |*dw, *mw, *vw| {
            mw.* = params.beta1 * mw.* + (1.0 - params.beta1) * dw.*;
            vw.* = params.beta2 * vw.* + (1.0 - params.beta2) * (dw.*) * (dw.*);
        }

        // Update the bias
        for (self.gradB, self.mB, self.vB) |*db, *mb, *vb| {
            mb.* = params.beta1 * mb.* + (1.0 - params.beta1) * db.*;
            vb.* = params.beta2 * vb.* + (1.0 - params.beta2) * (db.*) * (db.*);
        }
    }

    /// Applies the bias-corrected Adam update rule to adjust weights and biases
    fn updateWeights(self: *const NN, t: usize) void {
        // Transform t to float
        const tFloat: fType = @as(fType, @floatFromInt(t));

        // Compute the normalization for mt and vt
        const normM: fType = 1.0 - std.math.pow(fType, params.beta1, tFloat);
        const normV: fType = 1.0 - std.math.pow(fType, params.beta2, tFloat);

        // Update the weights
        for (self.weights, self.mW, self.vW) |*w, *mw, *vw| {
            w.* -= ((mw.*) / normM) * params.lr / (@sqrt(vw.* / normV) + params.eps);
        }

        // Update the bias
        for (self.biases, self.mB, self.vB) |*b, *mb, *vb| {
            b.* -= ((mb.*) / normM) * params.lr / (@sqrt(vb.* / normV) + params.eps);
        }
    }

    /// Trains the neural network using mini-batch gradient descent with the Adam optimizer. Splits data into training and validation sets, and records loss per epoch.
    pub fn train(self: *const NN, inputs: []const fType, outputs: []const fType) !void {
        const nIn: usize = params.nNeurons[0];
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];

        // Check the dimensions of the inputs
        if (inputs.len % nIn != 0) {
            std.debug.print("Incompatible dimension of inputs and number of neurons in layer 0!\n", .{});
            return err.incompatibleSizes;
        }

        // Check the dimensions of the outputs
        if (outputs.len % nOut != 0) {
            std.debug.print("Incompatible dimension of outputs and number of neurons in layer -1!\n", .{});
            return err.incompatibleSizes;
        }

        // Check the number of inputs and outputs
        if (inputs.len / nIn != outputs.len / nOut) {
            std.debug.print("Incompatible number of inputs and outputs!\n", .{});
            return err.incompatibleSizes;
        }

        // Zero the losses
        self.zeroLosses();
 
        // Alloc the dL array used in the computation of the loss of the validation
        var dL: []fType = &.{};
        if (self.allocator.alloc(fType, nOut)) |slice| {
            dL = slice;
        } else |_| {
            std.debug.print("Error while allocating the slice for dL!", .{});
            return err.backProp;
        }
        defer self.allocator.free(dL);
        
        // Compute the dimension of the data
        const nData: usize = inputs.len / nIn;
        
        // Compute the size of training and validation
        const nTrain: usize = @as(usize, @intFromFloat(@as(fType, @floatFromInt(nData)) * params.rTrain));
        const nValF: fType = @as(fType, @floatFromInt(nData)) * params.rVal;
        const nVal: usize = @as(usize, @intFromFloat(nValF));

        // Compute the number of batches
        const nBatches: usize = nTrain / params.batchSize;
        const nBatchesF: fType = @as(fType, @floatFromInt(nBatches));

        // Run over all epochs
        for (0..params.nEpochs) |epoch| {
            var lossEpoch: fType = 0.0;

            // Run over the batches
            for (0..nBatches) |batch| {
                // Compute the gradients
                if (self.backProp(inputs[batch * params.batchSize * nIn .. (batch + 1) * params.batchSize * nIn], outputs[batch * params.batchSize * nOut .. (batch + 1) * params.batchSize * nOut])) |lossE| {
                    lossEpoch += lossE / nBatchesF;
                } else |_| {
                    std.debug.print("Problem when trying to backprop in epoch {} and batch {}!\n", .{ epoch, batch });
                    return err.backProp;
                }

                // Compute mt and vt used by adam optimizer
                self.updateMV();

                self.updateWeights(epoch + 1);
            }

            // Save the training loss of this epoch
            self.lossesTraining[epoch] = lossEpoch;

            // Save the loss of the validation of this epoch
            self.lossesValidation[epoch] = 0.0;
            for (0..nVal) |i| {
                const pred: []fType = try self.nn.forward(inputs[(nTrain + i) * nIn .. (nTrain + i + 1) * nIn], &params.nNeurons, self.weights, self.biases, &params.activations);
                self.lossesValidation[epoch] += try loss.computeLoss(pred, outputs[(nTrain + i) * nOut .. (nTrain + i + 1) * nOut], dL, params.lossFunc) / nValF;
            }

            // Print the current state
            if (params.printEvery > 0 and epoch % params.printEvery == 0) {
                std.debug.print("Loss[{}] = ({}, {})\n", .{ epoch, self.lossesTraining[epoch], self.lossesValidation[epoch] });
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

        var inputs = try allocator.alloc(fType, nData * nIn);
        var outputs = try allocator.alloc(fType, nData * nOut);
        defer {
            allocator.free(inputs);
            allocator.free(outputs);
        }

        for (0..nData) |i| {
            for (0..nIn) |j| {
                inputs[i * nIn + j] = std.Random.floatNorm(rand, fType);
            }
            outputs[i * nOut] = 2.0 + 1.2 * inputs[i * nIn] - std.math.pow(fType, inputs[i * nIn + 1], 2) + @exp(-3.0 * inputs[i * nIn] - 2.0 * inputs[i * nIn + 1]);
            outputs[i * nOut + 1] = 1.4 + 3.0 * inputs[i * nIn] - std.math.pow(fType, inputs[i * nIn + 1], 2) + @exp(-2.0 * inputs[i * nIn] - 3.0 * inputs[i * nIn + 1]);
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

    /// Save the losses to a file
    pub fn saveLosses(self: *const NN, fileName: []const u8) !void {
        // Create a slice with both losses
        var losses: [2 * params.nEpochs]fType = undefined;

        // Save the losses of the training and validation
        for (0..params.nEpochs) |i| {
            losses[i] = self.lossesTraining[i];
            losses[params.nEpochs + i] = self.lossesValidation[i];
        }

        // Save the losses to a file
        try io.saveData(self.ioContext, fileName, &losses, params.nEpochs);
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
        var losses: [2 * params.nEpochs]fType = undefined;
        const dataDim = try io.loadData(ioContext, path, &losses);

        // Check the losses
        try testing.expectEqual(params.nEpochs, dataDim);
        try testing.expectEqual(nn.lossesTraining[0], losses[0]);
        try testing.expectEqual(nn.lossesValidation[0], losses[params.nEpochs]);

        // Delete the test file
        const cwd = std.Io.Dir.cwd();
        try cwd.deleteFile(ioContext, path);
    }
};

// Run the tests for the NN structure
comptime {
    std.testing.refAllDecls(@This());
}
