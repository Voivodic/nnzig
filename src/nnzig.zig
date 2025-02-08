const std = @import("std");
const core = @import("core");
const act = @import("act");
const loss = @import("loss");
const eigen = @import("eigen");
const mlp = @import("mlp");
const params = @import("params");

// Create the main structure for the NN
pub fn NN(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        nn: mlp.MLP(T),
        rand: std.rand,
        weights: []T = &.{},
        biases: []T = &.{},
        gradW: []T = &.{},
        gradB: []T = &.{},
        mW: []T = &.{},
        vW: []T = &.{},
        mB: []T = &.{},
        vB: []T = &.{},

        // Initialize the structure
        pub fn init(allocator: std.mem.Allocator) !NN(T) {
            // Check if the size of the nNeurons and activations is consistent
            if (params.activations.len != params.nNeurons.len - 1) {
                std.debug.print("Number of activation functions must be of the size of nNeurons - 1!\n", .{});
                return core.NNError.IncompatibleSizes;
            }

            // Initialize the random generator
            var xoshiro256 = std.Random.Xoshiro256.init(params.seed);

            // Create the NN
            var nn = NN(T){
                .allocator = allocator,
                .nn = try mlp.MLP(T).init(
                    allocator,
                    &params.nNeurons,
                ),
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
                std.debug.print("Failure when trying to allocate memory for the weights!\n", .{});
                return core.NNError.AllocationOfWeights;
            }

            // Alloc memory for the gradients of the weights
            if (allocator.alloc(T, nWeights)) |slice| {
                nn.gradW = slice;
            } else |_| {
                std.debug.print("Failure when trying to allocate memory for the gradient of the weights!\n", .{});
                return core.NNError.AllocationOfWeights;
            }

            // Alloc memory for the wt of the weights used in adam
            if (allocator.alloc(T, nWeights)) |slice| {
                nn.mW = slice;
            } else |_| {
                std.debug.print("Failure when trying to allocate memory for the mt of the weights!\n", .{});
                return core.NNError.AllocationOfWeights;
            }

            // Alloc memory for the vt of the weights used in adam
            if (allocator.alloc(T, nWeights)) |slice| {
                nn.vW = slice;
            } else |_| {
                std.debug.print("Failure when trying to allocate memory for the vt of the weights!\n", .{});
                return core.NNError.AllocationOfWeights;
            }

            // Alloc memory for the biases
            if (allocator.alloc(T, nBiases)) |slice| {
                nn.biases = slice;
            } else |_| {
                std.debug.print("Failure when trying to allocate memory for the biases!\n", .{});
                return core.NNError.AllocationOfBiases;
            }

            // Alloc memory for the gradients of the biases
            if (allocator.alloc(T, nBiases)) |slice| {
                nn.gradB = slice;
            } else |_| {
                std.debug.print("Failure when trying to allocate memory for the gradient of the biases!\n", .{});
                return core.NNError.AllocationOfBiases;
            }

            // Alloc memory for the wt of the biases used in adam
            if (allocator.alloc(T, nBiases)) |slice| {
                nn.mB = slice;
            } else |_| {
                std.debug.print("Failure when trying to allocate memory for the mt of the biases!\n", .{});
                return core.NNError.AllocationOfBiases;
            }

            // Alloc memory for the vt of the biases used in adam
            if (allocator.alloc(T, nBiases)) |slice| {
                nn.vB = slice;
            } else |_| {
                std.debug.print("Failure when trying to allocate memory for the vt of the biases!\n", .{});
                return core.NNError.AllocationOfBiases;
            }

            // Set the parameters to normal random numbers
            for (nn.weights) |*weight| {
                weight.* = std.Random.floatNorm(nn.rand, T);
            }
            for (nn.biases) |*bias| {
                bias.* = std.Random.floatNorm(nn.rand, T);
            }

            return nn;
        }

        // Deinitialize the NN
        pub fn deinit(self: *const NN(T)) void {
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
        }

        // Put the gradients equal to zero
        pub fn zeroGrad(self: *const NN(T)) void {
            eigen.setZero(self.gradW.ptr, self.gradW.len);
            eigen.setZero(self.gradB.ptr, self.gradB.len);
        }

        // Compute a forward pass in the NN
        pub fn forward(self: *const NN(T), input: []const T) ![]T {
            // Check the size of the input
            if (input.len != params.nNeurons[0]) {
                std.debug.print("The input has a size different of nNeurons[0]!\n", .{});
                return core.NNError.IncompatibleSizes;
            }

            // Pass the input trough the NN
            return self.nn.forward(input, &params.nNeurons, self.weights, self.biases, &params.activations);
        }

        // Compute the backpropagation given some data
        pub fn backProp(self: *const NN(T), inputs: [][]const T, outputs: [][]const T) !T {
            // Check the dimensions of the inputs
            if (inputs[0].len != params.nNeurons[0]) {
                std.debug.print("Incompatible dimension of inputs and number of neurons in layer 0!\n", .{});
                return core.NNError.IncompatibleSizes;
            }

            // Check the dimensions of the outputs
            if (outputs[0].len != params.nNeurons[params.nNeurons.len - 1]) {
                std.debug.print("Incompatible dimension of outputs and number of neurons in layer -1!\n", .{});
                return core.NNError.IncompatibleSizes;
            }

            // Check the number of inputs and outputs
            if (inputs.len != outputs.len) {
                std.debug.print("Incompatible number of inputs and outputs!\n", .{});
                return core.NNError.IncompatibleSizes;
            }

            // Get the data size
            const nData: usize = inputs.len;

            // Set the gradients to zero
            self.zeroGrad();

            // Slice to keep the predictions
            var pred: []T = undefined;
            if (self.allocator.alloc(T, outputs[0].len)) |slice| {
                pred = slice;
            } else |_| {
                std.debug.print("Problem to allocate the array for the predictions!\n", .{});
                return core.NNError.BackProp;
            }
            defer self.allocator.free(pred);

            // Slice to keep the derivatives of the loss
            var dL: []T = undefined;
            if (self.allocator.alloc(T, outputs[0].len)) |slice| {
                dL = slice;
            } else |_| {
                std.debug.print("Problem to allocate the array for the derivatives of the loss!\n", .{});
                return core.NNError.BackProp;
            }
            defer self.allocator.free(dL);

            // Save the value of the total loss
            var lossTotal: T = 0.0;

            // Run over all inputs and outputs
            for (inputs, outputs) |*in, *out| {
                // Compute the forward pass
                pred = self.nn.forward(in.*, &params.nNeurons, self.weights, self.biases, &params.activations);

                // Compute the loss and its derivative
                lossTotal += loss.computeLoss(T, pred, out.*, dL, params.lossFunc) / @as(T, @floatFromInt(nData));

                // Compute the gradient
                self.nn.backward(dL, &params.nNeurons, self.weights, self.biases, self.gradW, self.gradB);
            }

            // Return the total loss computed
            return lossTotal;
        }

        // Train the weights and biases using the adam optimizer
        pub fn train(self: *const NN(T), inputs: [][]const T, outputs: [][]const T) !void {
            // Check the dimensions of the inputs
            if (inputs[0].len != params.nNeurons[0]) {
                std.debug.print("Incompatible dimension of inputs and number of neurons in layer 0!\n", .{});
                return core.NNError.IncompatibleSizes;
            }

            // Check the dimensions of the outputs
            if (outputs[0].len != params.nNeurons[params.nNeurons.len - 1]) {
                std.debug.print("Incompatible dimension of outputs and number of neurons in layer -1!\n", .{});
                return core.NNError.IncompatibleSizes;
            }

            // Check the number of inputs and outputs
            if (inputs.len != outputs.len) {
                std.debug.print("Incompatible number of inputs and outputs!\n", .{});
                return core.NNError.IncompatibleSizes;
            }

            // Get the data size and number of batchs
            const nData: usize = inputs.len;
            const nBatches: usize = nData / params.batchSize;

            // Create a slice to keep the index of each input/output pair
            var inds: []usize = &.{};
            if (self.allocator.alloc(usize, nData)) |slice| {
                inds = slice;
            } else |_| {
                std.debug.print("Problem when trying to allocate the slice for the indeces!\n", .{});
            }
            defer self.allocator.free(inds);

            // Fill the inds with the index of each input/putput pair
            for (inds, 0..inds.len) |*ind, i| {
                ind.* = i;
            }

            // Run over all epochs
            for (0..params.nEpochs) |epoch| {
                _ = epoch;

                // Shuffle the data
                std.Random.shuffle(self.rand, usize, inds);

                // Run over all batchs
                for (0..nBatches) |batch| {
                    _ = batch;

                    // Set the gradients to zero
                    self.zeroGrad();
                }
            }
        }
    };
}
