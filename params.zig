const act = @import("act");
const loss = @import("loss");

// Set the number of neurons in each layer, including the input and output layers
// [Ninputs, Nhidden1, ..., NhiddenN, Noutputs]
pub const nNeurons = [_]usize{ 2, 16, 16, 2 };

// Set the activation function used in each layer
// The activativons are enums from act.activation
// It should have the sine of nNeurons - 1
const activation = act.Activation;
pub const activations = [_]activation{
    activation.relu,
    activation.relu,
    activation.none,
};

// Set the seed used for the pseudo random number generators
pub const seed: u64 = 12345;

// Set the parameters used by the adam optimizator
pub const beta1: f32 = 0.9;
pub const beta2: f32 = 0.999;
pub const lr: f32 = 0.001;
pub const nEpochs: usize = 100;
pub const batchSize: usize = 50;
pub const lossFunc: loss.losses = loss.losses.MSE;
