//! Main module that defines the main structure used by the library

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

    // Initialize the structure
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

    // Deinitialize the NN
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

    // Compute the normalization for the data
    pub fn computeNormalization(self: *const NN, inputs: [][]fType, outputs: [][]fType) !void {
        try self.norm.computeNormalization(inputs, outputs);
    }

    // Normalize the data
    pub fn normalize(self: *const NN, inputs: [][]fType, outputs: [][]fType) !void {
        try self.norm.normalize(inputs, outputs);
    }

    // Set the gradients to zero
    pub fn zeroGrad(self: *const NN) void {
        eigen.setZero(fType, self.gradW);
        eigen.setZero(fType, self.gradB);
    }

    // Set the losses to zero
    pub fn zeroLosses(self: *const NN) void {
        eigen.setZero(fType, self.lossesTraining);
        eigen.setZero(fType, self.lossesValidation);
    }

    // Compute a forward pass in the NN
    pub fn forward(self: *const NN, input: []const fType) ![]fType {
        // Check the size of the input
        if (input.len != params.nNeurons[0]) {
            std.debug.print("The input has a size different of nNeurons[0]!\n", .{});
            return err.incompatibleSizes;
        }

        // Pass the input trough the NN
        return self.nn.forward(input, &params.nNeurons, self.weights, self.biases, &params.activations);
    }

    // Compute the backpropagation given some data
    fn backProp(self: *const NN, inputs: [][]fType, outputs: [][]fType) !fType {
        // Get the data size
        const nData: fType = @as(fType, @floatFromInt(inputs.len));

        // Set the gradients to zero
        self.zeroGrad();

        // Slice to keep the derivatives of the loss
        var dL: []fType = undefined;
        if (self.allocator.alloc(fType, outputs[0].len)) |slice| {
            dL = slice;
        } else |_| {
            std.debug.print("Problem to allocate the array for the derivatives of the loss!\n", .{});
            return err.backProp;
        }
        defer self.allocator.free(dL);

        // Save the value of the total loss
        var lossTotal: fType = 0.0;

        // Run over all inputs and outputs
        for (inputs, outputs) |*in, *out| {
            // Compute the forward pass
            const pred: []fType = try self.nn.forward(in.*, &params.nNeurons, self.weights, self.biases, &params.activations);

            // Compute the loss and its derivative
            lossTotal += try loss.computeLoss(pred, out.*, dL, params.lossFunc) / nData;

            // Compute the gradient
            self.nn.backward(dL, &params.nNeurons, self.weights, self.biases, self.gradW, self.gradB);
        }

        // Return the total loss computed
        return lossTotal;
    }

    // Update the V and M used by adam
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

    // Update the weights and biases
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

    // Function used to train the NN
    pub fn train(self: *const NN, inputs: [][]fType, outputs: [][]fType) !void {
        // Check the dimensions of the inputs
        if (inputs[0].len != params.nNeurons[0]) {
            std.debug.print("Incompatible dimension of inputs and number of neurons in layer 0!\n", .{});
            return err.incompatibleSizes;
        }

        // Check the dimensions of the outputs
        if (outputs[0].len != params.nNeurons[params.nNeurons.len - 1]) {
            std.debug.print("Incompatible dimension of outputs and number of neurons in layer -1!\n", .{});
            return err.incompatibleSizes;
        }

        // Check the number of inputs and outputs
        if (inputs.len != outputs.len) {
            std.debug.print("Incompatible number of inputs and outputs!\n", .{});
            return err.incompatibleSizes;
        }
 
        // Alloc the dL array used in the computation of the loss of the validation
        var dL: []fType = &.{};
        if (self.allocator.alloc(fType, outputs[0].len)) |slice| {
            dL = slice;
        } else |_| {
            std.debug.print("Error while allocating the slice for dL!", .{});
            return err.backProp;
        }
        defer self.allocator.free(dL);

        // Compute the size of training and validation
        const nTrain: usize = @as(usize, @intFromFloat(@as(fType, @floatFromInt(inputs.len)) * params.rTrain));
        const nValF: fType = @as(fType, @floatFromInt(inputs.len)) * params.rVal;

        // Compute the number of batches
        const nBatches: usize = nTrain / params.batchSize;
        const nBatchesF: fType = @as(fType, @floatFromInt(nBatches));

        // Run over all epochs
        for (0..params.nEpochs) |epoch| {
            var lossEpoch: fType = 0.0;

            // Run over the batches
            for (0..nBatches) |batch| {
                // Compute the gradients
                if (self.backProp(inputs[batch * params.batchSize .. (batch + 1) * params.batchSize], outputs[batch * params.batchSize .. (batch + 1) * params.batchSize])) |lossE| {
                    lossEpoch += lossE / nBatchesF;
                } else |_| {
                    std.debug.print("Problem when trying to backprop in epoch {} and batch {}!\n", .{ epoch, batch });
                    return err.backProp;
                }

                // Compute mt and vt used by adam optimizer
                self.updateMV();

                // Update the weights and biases
                self.updateWeights(epoch + 1);
            }

            // Save the training loss of this epoch
            self.lossesTraining[epoch] = lossEpoch;

            // Save the loss of the validation of this epoch
            self.lossesValidation[epoch] = 0.0;
            for (inputs[nTrain..], outputs[nTrain..]) |*in, *out| {
                const pred: []fType = try self.nn.forward(in.*, &params.nNeurons, self.weights, self.biases, &params.activations);
                self.lossesValidation[epoch] += try loss.computeLoss(pred, out.*, dL, params.lossFunc) / nValF;
            }

            // Print the current state
            if (params.printEvery > 0 and epoch % params.printEvery == 0) {
                std.debug.print("Loss[{}] = ({}, {})\n", .{ epoch, self.lossesTraining[epoch], self.lossesValidation[epoch] });
            }
        }
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

