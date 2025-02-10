// Define the possible errors for the newtwork
pub const NNError = error{
    IncompatibleSizes,
    AllocationOfHiddens,
    AllocationOfWeights,
    AllocationOfBiases,
    InitNN,
    ForwardPass,
    BackProp,
};

//Define possible erros for handling the data
pub const DataError = error{
    InputAllocation,
    OutputAllocation,
    LossesAllocation,
    Normalization,
};
