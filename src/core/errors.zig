//! This module defines the error sets used throughout the nnzig library,
//! including errors for neural network operations, normalization, activations,
//! loss computation, and file I/O.

/// Error set for neural network operations
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

/// Error set for data normalization operations
pub const normalizationError = error{
    incompatibleSizes,
    aAllocation,
    bAllocation,
    notInitialized,
    threadRun,
};

/// Error set for activation function computation
pub const activationError = error{
    threadRun,
};

/// Error set for loss function computation
pub const lossError = error{
    threadRun,
};

/// Error set for binary file I/O operations
pub const ioError = error{
    precisionMismatch,
    invalidNNeurons,
    invalidNLayers,
    invalidNData,
};
