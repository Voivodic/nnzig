//! This module defines the error sets used throughout the nnzig library,
//! including errors for neural network operations, normalization, activations,
//! loss computation, and file I/O.

/// Error set for neural network operations, covering layer-size mismatches,
/// allocation failures, and the init, forward, and backward passes.
pub const nnError = error{
    incompatibleSizes,
    allocationOfHiddens,
    allocationOfWeights,
    allocationOfBiases,
    lossesAllocation,
    initNN,
    forwardPass,
    backProp,
};

/// Error set for data normalization operations, covering buffer allocation
/// failures, size mismatches, uninitialized state, and thread errors.
pub const normalizationError = error{
    incompatibleSizes,
    aAllocation,
    bAllocation,
    notInitialized,
    threadRun,
};

/// Error set for activation function computation, reporting thread execution errors.
pub const activationError = error{
    threadRun,
};

/// Error set for loss function computation, reporting thread execution errors.
pub const lossError = error{
    threadRun,
};

/// Error set for binary file I/O operations, covering precision, neuron-count,
/// layer-count, and data-count mismatches when loading or saving.
pub const ioError = error{
    precisionMismatch,
    invalidNNeurons,
    invalidNLayers,
    invalidNData,
};
