//! Main module of the nnzig library. Provides the `NN` structure which serves as the
//! primary interface for creating, training, saving, and loading neural networks.
//! Training uses mini-batch gradient descent with the Adam optimizer.
//! The `NN` struct delegates all layer-specific computation to the `MLP` layer,
//! acting as a thin orchestrator that can be extended with other layer types.

// Import the modules used
const std = @import("std");
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
    lossesTraining: []T = &.{},
    lossesValidation: []T = &.{},

    /// Initializes the neural network, allocating memory for the MLP layer, normalization data, and loss arrays
    pub fn init(allocator: std.mem.Allocator, ioContext: std.Io) !NN {
        // Initialize the random generator
        var xoshiro256 = std.Random.Xoshiro256.init(params.seed);

        // Create the NN — the MLP layer owns all weights, biases, gradients, and Adam moments
        var nn = NN{
            .allocator = allocator,
            .ioContext = ioContext,
            .nn = try mlp.MLP.init(allocator),
            .norm = try norms.Norm.init(allocator),
            .rand = std.Random.Xoshiro256.random(&xoshiro256),
        };

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

    /// Frees all memory allocated by `init`, including the MLP layer, normalization data, and loss arrays
    pub fn deinit(self: *const NN) void {
        // Deinit the MLP layer (frees weights, biases, gradients, Adam moments, activations)
        self.nn.deinit();

        // Free the memory used in the normalization
        self.norm.deinit();

        // Free the memory used in the losses array
        if (self.lossesTraining.len > 0) {
            self.allocator.free(self.lossesTraining);
            self.allocator.free(self.lossesValidation);
        }
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
        return self.nn.forward(input);
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

        // Single fixed sample: updateGrads then yields the per-sample gradient
        // (no batch averaging to reason about).
        var inputs = [_]T{ 0.5, -0.3 };
        var outputs = [_]T{ 1.0, 0.0 };
        var dL = [_]T{ 0.0, 0.0 };

        _ = try nn.nn.updateGrads(&inputs, &outputs);

        const eps: T = 1e-3;
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];
        const first_out_w: usize = nn.nn.weights.len - nOut * params.nNeurons[params.nNeurons.len - 2];
        const first_out_b: usize = nn.nn.biases.len - nOut;

        // Output layer uses the "none" activation, so perturbing its weights/
        // biases does not cross any ReLU kink -> finite differences are clean.
        const ow = nn.nn.weights[first_out_w];
        const aw = nn.nn.gradW[first_out_w];
        nn.nn.weights[first_out_w] = ow + eps;
        const lp_w = loss.computeLoss(try nn.forward(&inputs), &outputs, &dL, params.lossFunc);
        nn.nn.weights[first_out_w] = ow - eps;
        const lm_w = loss.computeLoss(try nn.forward(&inputs), &outputs, &dL, params.lossFunc);
        nn.nn.weights[first_out_w] = ow;
        const fd_w: T = (lp_w - lm_w) / (2 * eps);

        const ob = nn.nn.biases[first_out_b];
        const ab = nn.nn.gradB[first_out_b];
        nn.nn.biases[first_out_b] = ob + eps;
        const lp_b = loss.computeLoss(try nn.forward(&inputs), &outputs, &dL, params.lossFunc);
        nn.nn.biases[first_out_b] = ob - eps;
        const lm_b = loss.computeLoss(try nn.forward(&inputs), &outputs, &dL, params.lossFunc);
        nn.nn.biases[first_out_b] = ob;
        const fd_b: T = (lp_b - lm_b) / (2 * eps);

        std.log.info("gradCheck OUT: w analytic={} fd={} | b analytic={} fd={}", .{ aw, fd_w, ab, fd_b });

        try testing.expectApproxEqAbs(aw, fd_w, @abs(fd_w) * 1e-2 + 1e-4);
        try testing.expectApproxEqAbs(ab, fd_b, @abs(fd_b) * 1e-2 + 1e-4);
    }

    test "[nnzig] gradient descent consistency (all layers)" {
        // If gradW/gradB are the TRUE gradient of the mean batch loss, then a
        // small step w -= alpha*grad must reduce the loss by ~= alpha*||grad||^2.
        // This validates EVERY layer's backward pass (including the hidden-layer
        // delta propagation that the output-layer FD test above skips).
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        const ioContext = std.testing.io;
        var nn = try NN.init(allocator, ioContext);
        defer nn.deinit();

        const nIn: usize = params.nNeurons[0];
        const nOut: usize = params.nNeurons[params.nNeurons.len - 1];
        const nBatch: usize = 4;

        var inputs = [_]T{ 0.5, -0.3, 0.8, 0.2, -0.6, 0.9, 0.1, -0.7 };
        var outputs = [_]T{ 1.0, 0.0, 0.2, 0.7, -0.5, 0.4, 0.9, -0.2 };
        var dL = [_]T{ 0.0, 0.0 };
        _ = &dL;

        const loss0 = try nn.nn.updateGrads(&inputs, &outputs) / nBatch;

        // squared gradient norm (mean gradient, since updateGrads averages)
        var g2: T = 0.0;
        for (nn.nn.gradW) |g| g2 += g * g;
        for (nn.nn.gradB) |g| g2 += g * g;

        const alpha: T = 1e-4;
        for (nn.nn.weights, nn.nn.gradW) |*w, g| w.* -= alpha * g;
        for (nn.nn.biases, nn.nn.gradB) |*b, g| b.* -= alpha * g;

        // recompute mean batch loss
        var loss1: T = 0.0;
        for (0..nBatch) |i| {
            const pred = try nn.forward(inputs[i * nIn .. (i + 1) * nIn]);
            loss1 += loss.computeLoss(pred, outputs[i * nOut .. (i + 1) * nOut], &dL, params.lossFunc) / @as(T, @floatFromInt(nBatch));
        }

        const predicted_drop: T = alpha * g2;
        const actual_drop: T = loss0 - loss1;

        // ratio should be ~1.0 (within a few %); second-order error is O(alpha^2).
        try testing.expectApproxEqAbs(actual_drop, predicted_drop, @abs(predicted_drop) * 5e-2 + 1e-9);
    }

    test "[nnzig] gradient check vs finite differences (hidden layer 2)" {
        // Direct FD check on a SECOND-hidden-layer weight (4->4), which crosses
        // one ReLU. Picks a weight whose pre-activation is comfortably > 0 so
        // the small perturbation does not flip the ReLU sign.
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        const ioContext = std.testing.io;
        var nn = try NN.init(allocator, ioContext);
        defer nn.deinit();

        var inputs = [_]T{ 0.5, -0.3 };
        var outputs = [_]T{ 1.0, 0.0 };
        var dL = [_]T{ 0.0, 0.0 };

        _ = try nn.nn.updateGrads(&inputs, &outputs);

        // layer 2 weights start at index nNeurons[1]*nNeurons[2]... actually
        // layer1 = [0..8], layer2 = [8..24]. Probe several layer-2 weights and
        // pick the first whose FD is kink-free (central diff matches itself
        // under half-step -> monotone region).
        const eps: T = 1e-4;
        const layer2_start: usize = params.nNeurons[1] * params.nNeurons[2];
        const layer2_end: usize = layer2_start + params.nNeurons[2] * params.nNeurons[3];
        var checked: usize = 0;
        var max_rel_err: T = 0.0;
        for (layer2_start..layer2_end) |wi| {
            const ow = nn.nn.weights[wi];
            const aw = nn.nn.gradW[wi];
            nn.nn.weights[wi] = ow + eps;
            const lp = loss.computeLoss(try nn.forward(&inputs), &outputs, &dL, params.lossFunc);
            nn.nn.weights[wi] = ow - eps;
            const lm = loss.computeLoss(try nn.forward(&inputs), &outputs, &dL, params.lossFunc);
            nn.nn.weights[wi] = ow + eps / 2;
            const lp2 = loss.computeLoss(try nn.forward(&inputs), &outputs, &dL, params.lossFunc);
            nn.nn.weights[wi] = ow - eps / 2;
            const lm2 = loss.computeLoss(try nn.forward(&inputs), &outputs, &dL, params.lossFunc);
            nn.nn.weights[wi] = ow;
            const fd: T = (lp - lm) / (2 * eps);
            const fd2: T = (lp2 - lm2) / eps;
            // if the two FD estimates disagree, we are near a ReLU kink -> skip
            if (@abs(fd - fd2) > @abs(fd) * 1e-2 + 1e-6) continue;
            const rel = @abs(aw - fd) / (@abs(fd) + 1e-6);
            if (rel > max_rel_err) max_rel_err = rel;
            checked += 1;
        }

        try testing.expect(checked > 0);
        try testing.expect(max_rel_err < 1e-2);
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

    /// Trains the neural network using mini-batch gradient descent with the Adam optimizer. 
    /// Splits data into training and validation sets, and records loss per epoch.
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
        if (self.allocator.alloc(T, nOut * params.batchSizeCompute)) |slice| {
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
        const nTrainF: T = @as(T, @floatFromInt(nTrain));
        const nValF: T = @as(T, @floatFromInt(nData)) * params.rVal;
        const nVal: usize = @as(usize, @intFromFloat(nValF));

        // Compute the number of batches
        const nBatches: usize = nTrain / params.batchSize;
        const nRem: usize = nTrain % params.batchSize;
        const nBatchesVal: usize = nVal / params.batchSizeCompute;
        const nRemVal: usize = nVal % params.batchSizeCompute;

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
        // shuffled samples into contiguous buffers keeps updateGrads cache-friendly.
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
                if (self.nn.updateGrads(inputsBatch, outputsBatch)) |lossE| {
                    lossEpoch += lossE / nTrainF;
                } else |_| {
                    std.log.err("Problem when trying to backprop in epoch {} and batch {}!\n", .{ epoch, batch });
                    return err.backProp;
                }

                // Update the weights
                self.nn.updateWeights();
            }

            // Handle the remainder batch
            if (nRem != 0) {
                // Gather the remaining shuffled samples into the batch buffers
                for (0..nRem) |i| {
                    const idx = indexes[nBatches * params.batchSize + i];
                    @memcpy(inputsBatch[i * nIn .. (i + 1) * nIn], inputs[idx * nIn .. (idx + 1) * nIn]);
                    @memcpy(outputsBatch[i * nOut .. (i + 1) * nOut], outputs[idx * nOut .. (idx + 1) * nOut]);
                }

                // Compute the gradients
                if (self.nn.updateGrads(inputsBatch[0 .. nRem * nIn], outputsBatch[0 .. nRem * nOut])) |lossE| {
                    lossEpoch += lossE / nTrainF;
                } else |_| {
                    std.log.err("Problem when trying to backprop in epoch {} and batch {}!\n", .{ epoch, nBatches });
                    return err.backProp;
                }

                // Update the weights
                self.nn.updateWeights();
            }

            // Save the training loss of this epoch
            self.lossesTraining[epoch] = lossEpoch;

            // Save the loss of the validation of this epoch
            self.lossesValidation[epoch] = 0.0;
            for (0..nBatchesVal) |batch| {
                const pred: []T = try self.nn.forward(inputs[(nTrain + batch * params.batchSizeCompute) * nIn .. (nTrain + (batch + 1) * params.batchSizeCompute) * nIn]);
                self.lossesValidation[epoch] += loss.computeLoss(pred, outputs[(nTrain + batch * params.batchSizeCompute) * nOut .. (nTrain + (batch + 1) * params.batchSizeCompute) * nOut], dL, params.lossFunc) / nValF;
            }
            if (nRemVal > 0) {
                const pred: []T = try self.nn.forward(inputs[(nTrain + nBatchesVal * params.batchSizeCompute) * nIn .. (nTrain + nBatchesVal * params.batchSizeCompute + nRemVal) * nIn]);
                self.lossesValidation[epoch] += loss.computeLoss(pred, outputs[(nTrain + nBatchesVal * params.batchSizeCompute) * nOut .. (nTrain + nBatchesVal * params.batchSizeCompute + nRemVal) * nOut], dL[0 .. nRemVal * nOut], params.lossFunc) / nValF;
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
        nnIn.nn.weights[0] = 2.0;
        nnIn.nn.biases[0] = 1.0;

        // Save the weights to a file
        try nnIn.saveWeights(path);

        // Load the weights from the same file
        try nnOut.loadWeights(path);

        // Check the weights
        try testing.expectEqual(nnIn.nn.weights[0], nnOut.nn.weights[0]);
        try testing.expectEqual(nnIn.nn.biases[0], nnOut.nn.biases[0]);

        // Delete the test file
        const cwd = std.Io.Dir.cwd();
        try cwd.deleteFile(ioContext, path);
    }

    /// Save data points to a binary file
    pub fn saveData(self: *const NN, fileName: []const u8, dataIn: []const T, dataOut: []const T) !void {
        try io.saveData(self.ioContext, fileName, dataIn, dataOut, params.nNeurons[0], params.nNeurons[params.nNeurons.len - 1]);
    }

    /// Load data points from a binary file
    pub fn loadData(self: *const NN, fileName: []const u8) !struct { []T, []T } {
        const result = try io.loadData(self.allocator, self.ioContext, fileName);

        return result;
    }

    // Test the saving and loading of the data
    test "[nnzig] save/load-Data" {
        // Create the IO context
        const path = "test.bin";
        const ioContext = std.testing.io;

        // Create the allocator
        var gpa = std.heap.DebugAllocator(.{}){};
        const allocator = gpa.allocator();
        defer {
            const deinit_status = gpa.deinit();
            if (deinit_status == .leak) std.log.err("Leak!\n", .{});
        }

        // Initialize the neural network
        const nn = try NN.init(allocator, ioContext);
        defer nn.deinit();

        // Create some data points
        const nData: usize = 10;
        const nDim: usize = 4;
        var dataIn: [nData * nDim]T = undefined;
        var dataOut: [nData * nDim]T = undefined;
        for (0..nDim * nData) |i| {
            dataIn[i] = @as(T, @floatFromInt(i + 1));
            dataOut[i] = @as(T, @floatFromInt(3 * (i + 1)));
        }

        // Save the data points to a file
        try nn.saveData(path, &dataIn, &dataOut);

        // Load the weights from the same file
        const result = try nn.loadData(path);
        defer allocator.free(result[0]);
        defer allocator.free(result[1]);

        // Check the data
        for (0..nDim * nData) |i| {
            try testing.expectEqual(dataIn[i], result[0][i]);
            try testing.expectEqual(dataOut[i], result[1][i]);
        }

        // Delete the test file
        const cwd = std.Io.Dir.cwd();
        try cwd.deleteFile(ioContext, path);
    }

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
        const result = try io.loadData(allocator, ioContext, path);
        defer allocator.free(result[0]);
        defer allocator.free(result[1]);

        // Check the losses
        try testing.expectEqual(nn.lossesTraining[0], result[0][0]);
        try testing.expectEqual(nn.lossesValidation[0], result[1][0]);

        // Delete the test file
        const cwd = std.Io.Dir.cwd();
        try cwd.deleteFile(ioContext, path);
    }
};

// Run the tests for the NN structure
comptime {
    std.testing.refAllDecls(@This());
}
