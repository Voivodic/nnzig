//! This module parses compile-time parameters from a ZON configuration file
//! and exposes them as typed constants. It handles validation, type conversion,
//! and provides sensible defaults for optional fields.
const std = @import("std");
const paramsFile = @import("paramsFile");

/// Set the precision of the floats used
pub const T: type = if (@hasField(@TypeOf(paramsFile), "precision"))
    convertIntToType(paramsFile.precision)
else
    f32;

// Check if the number of neurons was given
const nLayers = blk: {
    if (@hasField(@TypeOf(paramsFile), "nNeurons")) {
        break :blk paramsFile.nNeurons.len;
    } else {
        break :blk 0;
    }
};

// --- Enums definitions ---

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

/// Enum of the possible ways to normalize the data
pub const norm = enum(u8) {
    meanStd,
};

// --- Define the base parameters ---

const baseParams = struct {
    numThreads: usize = 1,
    lossFunc: []const u8 = "MSE",
    normalization: []const u8 = "meanStd",
    seed: u64 = 12345,
    rTrain: T = 0.7,
    rVal: T = 0.2,
    eps: T = 1e-8,
    beta1: T = 0.9,
    beta2: T = 0.999,
    lr: T = 0.01,
    nEpochs: usize = 500,
    batchSize: usize = 50,
    printEvery: usize = 0,
};

// --- Conversion functions ---
 
/// Converts a compile-time int to its matching type
fn convertIntToType(comptime pre: usize) type {
    switch (pre) {
        16 => return f16,
        32 => return f32,
        64 => return f64,
        else => @compileError("Invalid precision!"),
    }
}

/// Converts a compile-time string to its matching Enum attribute
fn convertStringToEnum(comptime enum_type: type, comptime str: []const u8) enum_type {
    const maybe_enum = comptime std.meta.stringToEnum(enum_type, str);

    if (comptime maybe_enum == null) {
        @compileError("Value '" ++ str ++ "' is not a valid attribute of " ++ @typeName(enum_type));
    }

    return maybe_enum.?;
}

// Test the conversion to loss enum
test "[params] stringToEnum" {
    const fieldNameLoss = comptime std.meta.fieldNames(loss)[0];
    const result = convertStringToEnum(loss, fieldNameLoss);
    try std.testing.expectEqual(result, @field(loss, fieldNameLoss));

    const fieldNameNorm = comptime std.meta.fieldNames(norm)[0];
    const resultNorm = convertStringToEnum(norm, fieldNameNorm);
    try std.testing.expectEqual(resultNorm, @field(norm, fieldNameNorm));
}

/// Converts a compile-time ZON tuple into a standard fixed-size array
fn convertTupleToArray(comptime ElemType: type, comptime tuple: anytype) [tuple.len]ElemType {
    var arr: [tuple.len]ElemType = undefined;

    inline for (tuple, 0..) |item, i| {
        arr[i] = @as(ElemType, item);
    }

    return arr;
}

// Test the conversion to array
test "[params] tupleToArray" {
    const tuple = .{ 1, 2, 3, 4 };
    const result = convertTupleToArray(usize, tuple);
    try std.testing.expectEqual(result.len, tuple.len);
    try std.testing.expectEqual(@TypeOf(result), [tuple.len]usize);
}

/// Converts a compile-time ZON tuple into a standard array of Enum attributes
fn convertTupleToEnumArray(comptime EnumType: type, comptime tuple: anytype) [tuple.len]EnumType {
    var arr: [tuple.len]EnumType = undefined;

    inline for (tuple, 0..) |item, i| {
        arr[i] = convertStringToEnum(EnumType, item);
    }

    return arr;
}

// Test the conversion to enum array
test "[params] tupleToEnumArray" {
    const lossTuple = comptime .{std.meta.fieldNames(loss)[0], std.meta.fieldNames(loss)[0], std.meta.fieldNames(loss)[0]};
    const lossResult = convertTupleToEnumArray(loss, lossTuple);
    try std.testing.expectEqual(lossResult.len, lossTuple.len);
    try std.testing.expectEqual(@TypeOf(lossResult), [lossTuple.len]loss);

    const normTuple = comptime .{std.meta.fieldNames(norm)[0], std.meta.fieldNames(norm)[0]};
    const normResult = convertTupleToEnumArray(norm, normTuple);
    try std.testing.expectEqual(normResult.len, normTuple.len);
    try std.testing.expectEqual(@TypeOf(normResult), [normTuple.len]norm);
}

// --- Exposed parameters ---

/// Set the number of neurons in each layer, including the input and output layers
/// [Ninputs, Nhidden1, ..., NhiddenN, Noutputs]
pub const nNeurons = blk: {
    if (nLayers > 2) {
        break :blk convertTupleToArray(usize, paramsFile.nNeurons);
    } else if (nLayers == 1) {
        @compileError("nNeurons must have at least two elements (input and output layers)!");
    } else {
        @compileError("params.zon must have a 'nNeurons' field with at least two elements!");
    }
};

/// Set the activation function used in each layer
/// It should have the size of the nNeurons array - 1
pub const activations = blk: {
    if (nLayers < 2) {
        @compileError("There are not enough layers to set activations!");
    } else {
        var acts: [nLayers - 1]activation = undefined;

        // The user provided activation functions, so use them if they are valid
        if (@hasField(@TypeOf(paramsFile), "activations")) {
            const actsTmp = convertTupleToEnumArray(activation, paramsFile.activations);
            // The user provided less activation functions than the number of layers - 1
            if (paramsFile.activations.len < nLayers - 1) {
                for (0..actsTmp.len) |i| {
                    acts[i] = actsTmp[i];
                }

                for (actsTmp.len..acts.len) |i| {
                    acts[i] = activation.relu ;
                }

                acts[acts.len - 1] = actsTmp[actsTmp.len - 1];
            // The user provided more activation functions than the number of layers - 1
            } else if (paramsFile.activations.len > nLayers - 1) {
                for (0..acts.len) |i| {
                    acts[i] = actsTmp[i];
                }

                acts[acts.len - 1] = actsTmp[actsTmp.len - 1];
            // The user provided the correct number of activation functions
            } else {
                acts = actsTmp;
            }
        // The user did not provide activation functions, so use the default
        } else {
            for (0..acts.len) |i| {
                acts[i] = activation.relu;
            }
            acts[acts.len - 1] = activation.none;
        }

        break :blk acts;
    }
};

/// Set the final configuration from the ZON file and base parameters
pub const config = blk: {
    var cfg = baseParams{};

    for (std.meta.fieldNames(@TypeOf(cfg))) |fieldName| {
        if (@hasField(@TypeOf(paramsFile), fieldName)) {
            @field(cfg, fieldName) = @as(@TypeOf(@field(cfg, fieldName)), @field(paramsFile, fieldName));
        }
    }

    break :blk cfg;
};

/// Set the loss function used
pub const lossFunc: loss = convertStringToEnum(loss, paramsFile.lossFunc);

/// Set the normalization type used
pub const normalization: norm = convertStringToEnum(norm, paramsFile.normalization);

/// Set the number of threads to use on the cpu
pub const numThreads: usize = blk: {
    if (config.numThreads < 1) {
        @compileError("numThreads is the number of threads to use on the cpu. It must be greater than 0!");
    }
    break :blk config.numThreads;
};

/// Set the seed used for the pseudo random number generators
pub const seed: u64 = config.seed;

/// Set how all data will be splitted into
pub const rTrain: T = blk: {
    if (config.rTrain < 0.0 or config.rTrain > 1.0) {
        @compileError("rTrain is the fraction of data to use for training. It must be between 0.0 and 1.0!");
    }
    break :blk config.rTrain;
};
/// Set the fraction of data to use for validation
pub const rVal: T = blk: {
    if (config.rVal < 0.0 or config.rVal > 1.0) {
        @compileError("rVal is the fraction of data to use for validation. It must be between 0.0 and 1.0!");
    }
    break :blk config.rVal;
};
/// Set the fraction of data to use for testing
pub const rTest: T = blk: {
    if (1.0 - rTrain - rVal < 0.0) {
        @compileError("rTrain + rVal is the fraction of data to use for testing. It must be between 0.0 and 1.0!");
    }
    break :blk 1.0 - rTrain - rVal;
};

/// Set the parameters used by the adam optimizator
pub const eps: T = blk: {
    if (config.eps <= 0.0) {
        break :blk 1e-4;
    }
    break :blk config.eps;
};
pub const beta1: T = blk: {
    if (config.beta1 < 0.0 or config.beta1 > 1.0) {
        break :blk 0.9;
    }
    break :blk config.beta1;
};
pub const beta2: T = blk: {
    if (config.beta2 < 0.0 or config.beta2 > 1.0) {
        break :blk 0.999;
    }
    break :blk config.beta2;
};
pub const lr: T = blk: {
    if (config.lr <= 0.0) {
        break :blk 0.01;
    }
    break :blk config.lr;
};

/// Set the number of epochs used by the adam optim
pub const nEpochs: usize = blk: {
    if (config.nEpochs < 1) {
        @compileError("nEpochs is the number of epochs to use for training. It must be greater than 0!");
    }
    break :blk config.nEpochs;
};

/// Set the batch size
pub const batchSize: usize = blk: {
    if (config.batchSize < 0) {
        @compileError("batchSize is the number of samples to use for each batch. It must be positive! Use 0 to use all samples in a batch.");
    }
    break :blk config.batchSize;
};

/// Set the frequency of printing results
pub const printEvery: usize = blk: {
    if (config.printEvery < 0) {
        @compileError("printEvery is the number of epochs to wait before printing results. It must be positive! Use 0 to not print results.");
    }
    break :blk config.printEvery;
};
