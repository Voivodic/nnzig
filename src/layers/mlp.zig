//! This module defines a simple multi layer perceptron (MPL)

// Import the used modules
const std = @import("std");
const eigen = @import("eigen");
const act = @import("act");
const params = @import("params");
const err = @import("errors").nnError;
const fType = params.floatType;

/// Define the structure for the MLP
pub const MLP = struct {
    allocator: std.mem.Allocator,
    y: []fType = &.{},
    dy: []fType = &.{},
    V: []fType = &.{},

    // Initialize the mlp
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
        if (allocator.alloc(fType, nHidden)) |slice| {
            mlp.y = slice;
        } else |_| {
            std.debug.print("Failure when trying to allocate memory for y!\n", .{});
            return err.allocationOfHiddens;
        }

        // Alloc space for the derivative of hidden values
        if (allocator.alloc(fType, nHidden)) |slice| {
            mlp.dy = slice;
        } else |_| {
            std.debug.print("Failure when trying to allocate memory for dy!\n", .{});
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
        if (allocator.alloc(fType, vSize)) |slice| {
            mlp.V = slice;
        } else |_| {
            std.debug.print("Failure when trying to allocate memory for V!\n", .{});
            return err.allocationOfHiddens;
        }

        return mlp;
    }

    // Free the memory allocated
    pub fn deinit(self: *const MLP) void {
        // Free V
        self.allocator.free(self.V);

        // Free the memory allocated for the hidden values and its derivatives
        self.allocator.free(self.y);
        self.allocator.free(self.dy);
    }

    // Compute the output of the NN to a single input
    pub fn forward(self: *const MLP, input: []const fType, nNeurons: []const usize, weights: []const fType, biases: []const fType, activations: []const params.activation) ![]fType {
        // Save the input in first hidden values
        for (input, 0..input.len) |*in, i| {
            self.y[i] = in.*;
            self.dy[i] = 0.0;
        }

        // Run over all layers and compute the partial retult
        var nprod: usize = 0;
        var nsum: usize = 0;
        for (0..(nNeurons.len - 1)) |i| {
            const nin = nNeurons[i];
            const nout = nNeurons[i + 1];

            // Compute the matrix vector multiplication then add the result to a third vector
            eigen.matrixVectorMulAdd(fType, weights[nprod..(nprod + nin * nout)], self.y[nsum..(nsum + nin)], biases[nsum..(nsum + nout)], self.y[(nsum + nin)..(nsum + nin + nout)]);

            // Apply the activation function
            try act.activateElements(self.y[(nsum + nin)..(nsum + nin + nout)], self.dy[(nsum + nin)..(nsum + nin + nout)], activations[i]);

            // Update the total sum and sum of products to keep track of the current position of the 1D slices
            nsum += nin;
            nprod += nin * nout;
        }

        // Return the last slice of hidden values (the output)
        return self.y[nsum..];
    }

    // Compute the NN backward and update the parameters
    pub fn backward(self: *const MLP, dL: []const fType, nNeurons: []const usize, weights: []const fType, biases: []const fType, gradW: []fType, gradB: []fType) void {
        // Counters used to iterate over the weights and biases
        var layer: usize = nNeurons.len - 1;
        var nwIni: usize = weights.len - nNeurons[layer] * nNeurons[layer - 1];
        var nwEnd: usize = weights.len;
        var nbIni: usize = biases.len - nNeurons[layer];
        var nbEnd: usize = biases.len;
        var nyIni: usize = self.y.len - nNeurons[layer];
        var nyEnd: usize = self.y.len;

        // Compute the initial value for the vector V
        eigen.vectorInit(fType, dL, self.V);

        // Iterate from the last to the first layer
        while (layer > 0) : (layer -= 1) {
            // Multiply the propagated vector V by the derivative of the activation function
            eigen.vectorMul(fType, self.V[0..(nNeurons[layer])], self.dy[nyIni..nyEnd]);

            // Update the gradient for the weights
            eigen.updateGradWeights(fType, self.V[0..nNeurons[layer]], self.y[(nyIni - nNeurons[layer - 1])..nyIni], gradW[nwIni..nwEnd]);

            // Update the gradient for the biases
            eigen.updateGradBiases(fType, self.V[0..nNeurons[layer]], gradB[nbIni..nbEnd]);

            // Do not run this part in the last iteration
            if (layer > 1) {

                // Update the M matrix using the current weight matrix
                eigen.vectorMatrixMul(fType, self.V[0..(nNeurons[layer])], weights[nwIni..nwEnd]);

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
};
