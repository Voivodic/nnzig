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
