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
        muIn: []T = &.{},
        stdIn: []T = &.{},
        muOut: []T = &.{},
        stdOut: []T = &.{},
        lossesTraining: []T = &.{},
        lossesValidation: []T = &.{},

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

            // Set mW and vW to zero
            eigen.setZero(nn.mW.ptr, nn.mW.len);
            eigen.setZero(nn.vW.ptr, nn.vW.len);

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

            // Set mb and vb to zero
            eigen.setZero(nn.mB.ptr, nn.mB.len);
            eigen.setZero(nn.vB.ptr, nn.vB.len);

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

            // Free the memory used in the normalization
            if (self.muIn.len > 0) {
                self.allocator.free(self.muIn);
                self.allocator.free(self.stdIn);
                self.allocator.free(self.muOut);
                self.allocator.free(self.stdOut);
            }

            // Free the memory used in the losses array
            if (self.lossesTraining.len > 0) {
                self.allocator.free(self.lossesTraining);
                self.allocator.free(self.lossesValidation);
            }
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
        fn backProp(self: *const NN(T), inputs: [][]const T, outputs: [][]const T) !T {
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
            const nData: T = @as(T, @floatFromInt(inputs.len));

            // Set the gradients to zero
            self.zeroGrad();

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
                const pred: []T = self.nn.forward(in.*, &params.nNeurons, self.weights, self.biases, &params.activations);

                // Compute the loss and its derivative
                lossTotal += loss.computeLoss(T, pred, out.*, dL, params.lossFunc) / nData;

                // Compute the gradient
                self.nn.backward(dL, &params.nNeurons, self.weights, self.biases, self.gradW, self.gradB);
            }

            // Return the total loss computed
            return lossTotal;
        }

        // Update the V and M used by adam
        fn updateMV(self: *NN(T)) void {
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
        fn updateWeights(self: *NN(T), t: usize) void {
            // Transform t to float
            const tFloat: T = @as(T, @floatFromInt(t));

            // Compute the normalization for mt and vt
            const normM: T = 1.0 - std.math.pow(T, params.beta1, tFloat);
            const normV: T = 1.0 - std.math.pow(T, params.beta2, tFloat);

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
        pub fn train(self: *NN(T), inputs: [][]T, outputs: [][]T) !void {
            // Create the slice for the losses of the training
            if (self.allocator.alloc(T, params.nEpochs)) |slice| {
                self.lossesTraining = slice;
            } else |_| {
                std.debug.print("Error while allocating the slice for the losses of the training!\n", .{});
                return core.DataError.LossesAllocation;
            }

            // Create the slice for the losses of the validation
            if (self.allocator.alloc(T, params.nEpochs)) |slice| {
                self.lossesValidation = slice;
            } else |_| {
                std.debug.print("Error while allocating the slice for the losses of the validation!\n", .{});
                return core.DataError.LossesAllocation;
            }

            // Alloc the dL array used in the computation of the loss of the validation
            var dL: []T = &.{};
            if (self.allocator.alloc(T, outputs[0].len)) |slice| {
                dL = slice;
            } else |_| {
                std.debug.print("Error while allocating the slice for dL!", .{});
                return core.NNError.BackProp;
            }
            defer self.allocator.free(dL);

            // Compute the size of training and validation
            const nTrain: usize = @as(usize, @intFromFloat(@as(T, @floatFromInt(inputs.len)) * params.nTrain));
            const nValF: T = @as(T, @floatFromInt(inputs.len)) * params.nVal;

            // Compute the number of batches
            const nBatches: usize = nTrain / params.batchSize;
            const nBatchesF: T = @as(T, @floatFromInt(nBatches));

            // Keep the current loss for each epoch
            var lossEpoch: T = 0.0;

            // Run over all epochs
            for (0..params.nEpochs) |epoch| {
                lossEpoch = 0.0;

                // Run over the batches
                for (0..nBatches) |batch| {
                    // Compute the gradients
                    if (self.backProp(inputs[batch * params.batchSize .. (batch + 1) * params.batchSize], outputs[batch * params.batchSize .. (batch + 1) * params.batchSize])) |lossE| {
                        lossEpoch += lossE / nBatchesF;
                    } else |_| {
                        std.debug.print("Problem when trying to backprop in epoch {} and batch {}!\n", .{ epoch, batch });
                        return core.NNError.BackProp;
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
                    const pred: []T = self.nn.forward(in.*, &params.nNeurons, self.weights, self.biases, &params.activations);
                    self.lossesValidation[epoch] += loss.computeLoss(T, pred, out.*, dL, params.lossFunc) / nValF;
                }

                // Print the current state
                if (epoch % params.printEvery == 0) {
                    try std.io.getStdOut().writer().print("Loss[{}] = ({}, {})\n", .{ epoch, self.lossesTraining[epoch], self.lossesValidation[epoch] });
                }
            }
        }

        // Normalize the data
        pub fn computeNormalization(self: *NN(T), inputs: [][]T, outputs: [][]T) !void {
            // Check if the size of inputs and outputs are the same
            if (inputs.len != outputs.len) {
                std.debug.print("Inputs and outputs size must be the same!\n", .{});
                return core.NNError.IncompatibleSizes;
            }

            // Alloc the slices for the mean and std of inputs and outputs
            if (self.allocator.alloc(T, inputs[0].len)) |slice| {
                self.muIn = slice;
            } else |_| {
                std.debug.print("Problem in the allocation of the mean of inputs!\n", .{});
                return core.DataError.InputAllocation;
            }

            // Allocate the slice for the std of the input
            if (self.allocator.alloc(T, inputs[0].len)) |slice| {
                self.stdIn = slice;
            } else |_| {
                std.debug.print("Problem in the allocation of the std of inputs!\n", .{});
                return core.DataError.InputAllocation;
            }

            // Zero the input arrays
            for (self.muIn, self.stdIn) |*mu, *st| {
                mu.* = 0.0;
                st.* = 0.0;
            }

            // Allocate the slice for the mean of the outputs
            if (self.allocator.alloc(T, outputs[0].len)) |slice| {
                self.muOut = slice;
            } else |_| {
                std.debug.print("Problem in the allocation of the mean of outputs!\n", .{});
                return core.DataError.OutputAllocation;
            }

            // Allocate the slice for the std of the outputs
            if (self.allocator.alloc(T, outputs[0].len)) |slice| {
                self.stdOut = slice;
            } else |_| {
                std.debug.print("Problem in the allocation of the std of outputs!\n", .{});
                return core.DataError.OutputAllocation;
            }

            // Zero the outputs arrys
            for (self.muOut, self.stdOut) |*mu, *st| {
                mu.* = 0.0;
                st.* = 0.0;
            }

            // Get the number of data points
            const N = inputs.len;

            // Compute the means
            for (inputs, outputs) |*ins, *outs| {
                for (ins.*, self.muIn) |*in, *mu| {
                    mu.* += in.* / @as(T, @floatFromInt(N));
                }
                for (outs.*, self.muOut) |*out, *mu| {
                    mu.* += out.* / @as(T, @floatFromInt(N));
                }
            }

            // Compute the stds
            for (inputs, outputs) |*ins, *outs| {
                for (ins.*, self.muIn, self.stdIn) |*in, *mu, *st| {
                    st.* += (in.* - mu.*) * (in.* - mu.*) / @as(T, @floatFromInt(N));
                }
                for (outs.*, self.muOut, self.stdOut) |*out, *mu, *st| {
                    st.* += (out.* - mu.*) * (out.* - mu.*) / @as(T, @floatFromInt(N));
                }
            }
        }

        // Normalize the slice
        pub fn normalize(self: NN(T), inputs: [][]T, outputs: [][]T) !void {
            // Check if the size of inputs and outputs are the same
            if (inputs.len != outputs.len) {
                std.debug.print("Inputs and outputs size must be the same!\n", .{});
                return core.NNError.IncompatibleSizes;
            }

            // Check if the normalizations were computed
            if (self.muIn.len == 0) {
                std.debug.print("The normalizations were not computed!\n", .{});
                return core.DataError.Normalization;
            }

            // Normalize the inputs and outputs
            for (inputs, outputs) |*ins, *outs| {
                for (ins.*, self.muIn, self.stdIn) |*in, *mu, *st| {
                    in.* = (in.* - mu.*) / st.*;
                }
                for (outs.*, self.muOut, self.stdOut) |*out, *mu, *st| {
                    out.* = (out.* - mu.*) / st.*;
                }
            }
        }

        // deNormalize the slice
        pub fn deNormalize(self: NN(T), inputs: [][]T, outputs: [][]T) !void {
            // Check if the size of inputs and outputs are the same
            if (inputs.len != outputs.len) {
                std.debug.print("Inputs and outputs size must be the same!\n", .{});
                return core.NNError.IncompatibleSizes;
            }

            // Check if the normalizations were computed
            if (self.muIn.len == 0) {
                std.debug.print("The normalizations were not computed!\n", .{});
                return core.DataError.Normalization;
            }

            // Normalize the inputs and outputs
            for (inputs, outputs) |*ins, *outs| {
                for (ins.*, self.muIn, self.stdIn) |*in, *mu, *st| {
                    in.* = (in.*) * (st.*) + mu.*;
                }
                for (outs.*, self.muOut, self.stdOut) |*out, *mu, *st| {
                    out.* = (out.*) * st.* + mu.*;
                }
            }
        }
    };
}
