//! This module sets all the parameters of the neural network at comptime

/// Enum of the possible activation functions
pub const activation = enum(u8) {
    none,
    relu,
    tanh,
    sigmoid,
};

/// Enum of the possible loss functions
pub const loss = enum(u8) {
    MSE,
};

/// Set the number of threads to use on the cpu
pub const numThreads: usize = 1;

/// Enum of the possible ways to normalize the data
pub const normType = enum(u8) {
    meanStd,
};

/// Set the number of neurons in each layer, including the input and output layers
/// [Ninputs, Nhidden1, ..., NhiddenN, Noutputs]
pub const nNeurons = [4]usize{ 2, 16, 16, 2 };

/// Set the activation function used in each layer
/// It should have the size of the nNeurons array - 1
pub const activations = [3]activation{
    activation.relu,
    activation.relu,
    activation.none,
};

/// Set the loss function used
pub const lossFunc: loss = loss.MSE;

/// Set the seed used for the pseudo random number generators
pub const seed: u64 = 12345;

/// Set how all data will be splitted into
pub const rTrain: f32 = 0.7;
pub const rVal: f32 = 0.2;
pub const rTest: f32 = 1.0 - rTrain - rVal;

/// Set the parameters used by the adam optimizator
pub const eps: f32 = 1e-8;
pub const beta1: f32 = 0.9;
pub const beta2: f32 = 0.999;
pub const lr: f32 = 0.01;

/// Set the number of epochs used by the adam optim
pub const nEpochs: usize = 500;

/// Set the batch size
pub const batchSize: usize = 50;

/// Set the frequency of printing results
pub const printEvery: usize = 50;
