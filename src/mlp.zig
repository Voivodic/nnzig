const std = @import("std");
const core = @import("core");
const eigen = @import("eigen");
const act = @import("act");

// Define the structure for the MLP
pub fn MLP(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        y: []T = &.{},
        dy: []T = &.{},
        V: []T = &.{},

        // Initialize the mlp
        pub fn init(allocator: std.mem.Allocator, nNeurons: []const usize) !MLP(T) {
            // Create the mlp
            var mlp = MLP(T){
                .allocator = allocator,
            };

            // Compute the total number of hidden values
            var nHidden: usize = 0;
            for (0..nNeurons.len) |i| {
                nHidden += nNeurons[i];
            }

            // Alloc space for the hidden values
            if (allocator.alloc(T, nHidden)) |slice| {
                mlp.y = slice;
            } else |_| {
                std.debug.print("Failure when trying to allocate memory for y!\n", .{});
                return core.NNError.AllocationOfHiddens;
            }

            // Alloc space for the derivative of hidden values
            if (allocator.alloc(T, nHidden)) |slice| {
                mlp.dy = slice;
            } else |_| {
                std.debug.print("Failure when trying to allocate memory for dy!\n", .{});
                return core.NNError.AllocationOfHiddens;
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
                std.debug.print("Failure when trying to allocate memory for V!\n", .{});
                return core.NNError.AllocationOfHiddens;
            }

            return mlp;
        }

        // Free the memory allocated
        pub fn deinit(self: *const MLP(T)) void {
            // Free V
            self.allocator.free(self.V);

            // Free the memory allocated for the hidden values and its derivatives
            self.allocator.free(self.y);
            self.allocator.free(self.dy);
        }

        // Compute the output of the NN to a single input
        pub fn forward(self: *const MLP(T), input: []const T, nNeurons: []const usize, weights: []const T, biases: []const T, activations: []const act.Activation) []T {
            // Save the input in first hidden values
            for (input, 0..input.len) |in, i| {
                self.y[i] = in;
                self.dy[i] = 0.0;
            }

            // Run over all layers and compute the partial retult
            var nprod: usize = 0;
            var nsum: usize = 0;
            for (0..(nNeurons.len - 1)) |i| {
                const nin = nNeurons[i];
                const nout = nNeurons[i + 1];

                // Compute the matrix vector multiplication then add the result to a third vector
                eigen.matrixVectorMulAdd(weights[nprod..(nprod + nin * nout)].ptr, self.y[nsum..(nsum + nin)].ptr, biases[nsum..(nsum + nout)].ptr, self.y[(nsum + nin)..(nsum + nin + nout)].ptr, nout, nin);

                // Apply the activation function
                act.activateElements(T, self.y[(nsum + nin)..(nsum + nin + nout)], self.dy[(nsum + nin)..(nsum + nin + nout)], activations[i]);

                // Update the total sum and sum of products to keep track of the current position of the 1D slices
                nsum += nin;
                nprod += nin * nout;
            }

            // Return the last slice of hidden values (the output)
            return self.y[nsum..];
        }

        // Compute the NN backward and update the parameters
        pub fn backward(self: *const MLP(T), dL: []const T, nNeurons: []const usize, weights: []const T, biases: []const T, gradW: []T, gradB: []T) void {
            // Counters used to iterate over the weights and biases
            var layer: usize = nNeurons.len - 1;
            var nwIni: usize = weights.len - nNeurons[layer] * nNeurons[layer - 1];
            var nwEnd: usize = weights.len;
            var nbIni: usize = biases.len - nNeurons[layer];
            var nbEnd: usize = biases.len;

            // Compute the initial value for the vector V
            eigen.vectorVInit(dL.ptr, self.V.ptr, dL.len);

            // Iterate from the last to the first layer
            while (layer > 0) : (layer -= 1) {
                std.debug.print("{}\n", .{layer});
                // Multiply the propagated vector V by the derivative of the activation function
                eigen.vectorMul(self.V[0..(nNeurons[layer])].ptr, self.dy[nbIni..nbEnd].ptr, nNeurons[layer]);
                std.debug.print("{}\n", .{layer});
                // Update the gradient for the weights
                std.debug.print("{} {} {} {} {} {}\n", .{ nNeurons[layer - 1], nNeurons[layer], nbIni, nbEnd, nwIni, nwEnd });
                eigen.updateGradWeights(self.V[0..nNeurons[layer]].ptr, self.y[(nbIni - nNeurons[layer - 1])..nbIni].ptr, gradW[nwIni..nwEnd].ptr, nNeurons[layer], nNeurons[layer - 1]);
                std.debug.print("{}\n", .{layer});
                // Update the gradient for the biases
                eigen.updateGradBiases(self.V[0..nNeurons[layer]].ptr, gradB[nbIni..nbEnd].ptr, nNeurons[layer]);
                std.debug.print("{}\n", .{layer});
                // Do not run this part in the last iteration
                if (layer > 1) {
                    std.debug.print("{}\n", .{layer});
                    // Update the M matrix using the current weight matrix
                    eigen.vectorMatrixMul(self.V[0..(nNeurons[layer])].ptr, weights[nwIni..nwEnd].ptr, nNeurons[layer], nNeurons[layer - 1]);
                    std.debug.print("{}\n", .{layer});
                    // Update the counters
                    nbEnd = nbIni;
                    nbIni -= nNeurons[layer - 1];
                    nwEnd = nwIni;
                    nwIni -= nNeurons[layer - 1] * nNeurons[layer - 2];
                }
            }
            std.debug.print("Done!\n", .{});
        }
    };
}
