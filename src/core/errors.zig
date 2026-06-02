//! This module defines the erros that might happen in the library

/// Define the possible errors for the newtwork
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

/// Possible errors for the normalization process
pub const normalizationError = error{
    incompatibleSizes,
    aAllocation,
    bAllocation,
    notInitialized,
    threadRun,
};

/// Possible erros for the computation of activation functions
pub const activationError = error{
    threadRun,
};

/// Possible erros for the computation of the loss function
pub const lossError = error{
    threadRun,
};

/// Possible errors for the IO module
pub const ioError = error{
    precisionMismatch,
    invalidNNeurons,
    invalidNLayers,
    invalidNData,
};
