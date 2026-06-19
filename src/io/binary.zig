//! Binary file I/O for neural network weights and generic data arrays.
//! Uses a simple header-based format to store precision, layer dimensions,
//! and raw weight/bias data for persistence across sessions.

// Import the modules
const std = @import("std");
const nnzig = @import("nnzig");
const params = @import("params");
const errors = @import("errors");
const err = errors.ioError;
const testing = std.testing;
const T = params.T;

/// Header structure for neural network binary files, storing precision and layer topology.
/// It is written before the raw weight and bias bytes so that files are self-describing.
const NNHeader = struct {
    precision: u64 = @sizeOf(T),
    nLayers: u64 = @as(u64, params.nNeurons.len),
    nNeurons: [params.nNeurons.len]u64 = @as([params.nNeurons.len]u64, params.nNeurons),
};

/// Header structure for generic data binary files, storing precision, count, and dimension.
const DataHeader = struct {
    precision: u64 = @sizeOf(T),
    nData: u64 = 0,
    dimIn: u64 = 0,
    dimOut: u64 = 0,
};

/// Saves the weights of the neural network to a binary file. Writes the `NNHeader`
/// followed by the raw weight and bias bytes.
pub fn saveWeights(fileName: []const u8, nn: *const nnzig.NN) !void {
    // Open the file in fileName
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.createFile(nn.ioContext, fileName, .{});
    defer file.close(nn.ioContext);

    // Create the header with standard values
    const header = NNHeader{};

    // Write the header to the file
    try file.writeStreamingAll(nn.ioContext, std.mem.asBytes(&header));

    // Write the weights and biases to the file
    try file.writeStreamingAll(nn.ioContext, std.mem.sliceAsBytes(nn.nn.weights));
    try file.writeStreamingAll(nn.ioContext, std.mem.sliceAsBytes(nn.nn.biases));
}

/// Loads the weights of the neural network from a binary file. Validates the `NNHeader`
/// against the current config and returns `ioError` on mismatch before reading the weight
/// and bias bytes.
pub fn loadWeights(fileName: []const u8, nn: *const nnzig.NN) !void {
    // Open the file in fileName
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(nn.ioContext, fileName, .{});
    defer file.close(nn.ioContext);

    // Create the buffer to read the header
    var read_buffer: [4096]u8 = undefined;
    var file_reader = file.reader(nn.ioContext, &read_buffer);
    var reader = &file_reader.interface;

    // Read the header from the file
    var header: NNHeader = undefined;
    try reader.readSliceAll(std.mem.asBytes(&header));

    // Check the header values
    if (header.precision != @sizeOf(T)) {
        std.log.err("Precision mismatch! Expected {} bits, got {} bits!", .{ @sizeOf(T), header.precision });
        return err.precisionMismatch;
    }
    if (header.nLayers != params.nNeurons.len) {
        std.log.err("Number of layers mismatch! Expected {}, got {}!", .{ params.nNeurons.len, header.nLayers });
        return err.invalidNLayers;
    }
    for (header.nNeurons, 0..) |n, i| {
        if (n != params.nNeurons[i]) {
            std.log.err("Number of neurons in layer {} mismatch! Expected {}, got {}!", .{ i, params.nNeurons[i], n });
            return err.invalidNNeurons;
        }
    }

    // Read the weights and biases from the file
    try reader.readSliceAll(std.mem.sliceAsBytes(nn.nn.weights));
    try reader.readSliceAll(std.mem.sliceAsBytes(nn.nn.biases));
}

// Test the saving and loading of the weights
test "[io] save/load-Weights" {
    // Create the allocator
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) std.log.err("Leak!\n", .{});
    }

    // Create the IO context
    const path = "test.bin";
    const io = std.testing.io;

    // Create the neural network
    const nnIn = try nnzig.NN.init(allocator, io);
    defer nnIn.deinit();
    const nnOut = try nnzig.NN.init(allocator, io);
    defer nnOut.deinit();

    // Change some weights
    nnIn.nn.weights[0] = 2.0;
    nnIn.nn.biases[0] = 1.0;

    // Save the weights to a file
    try saveWeights(path, &nnIn);

    // Load the weights from the same file
    try loadWeights(path, &nnOut);

    // Check the weights
    try testing.expectEqual(nnIn.nn.weights[0], nnOut.nn.weights[0]);
    try testing.expectEqual(nnIn.nn.biases[0], nnOut.nn.biases[0]);

    // Delete the test file
    const cwd = std.Io.Dir.cwd();
    cwd.deleteFile(io, path) catch {};
}

/// Saves the data points to a binary file. Writes the `DataHeader` followed by the
/// input and output arrays.
pub fn saveData(io: std.Io, fileName: []const u8, dataIn: []const T, dataOut: []const T, dimIn: u64, dimOut: u64) !void {
    // Open the file in fileName
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.createFile(io, fileName, .{});
    defer file.close(io);

    // Check the size of the data
    if (dataIn.len % dimIn != 0) {
        std.log.err("The size of the data ({}) is not a multiple of dimData ({})!\n", .{ dataIn.len, dimIn });
        return err.invalidNData;
    }

    // Check the size of the data
    if (dataOut.len % dimOut != 0) {
        std.log.err("The size of the data ({}) is not a multiple of dimData ({})!\n", .{ dataOut.len, dimOut });
        return err.invalidNData;
    }

    // Check the sizes of the data
    if (dataIn.len / dimIn != dataOut.len / dimOut) {
        std.log.err("The size of the dataIn ({}) is not equal to the size of the dataOut ({})!\n", .{ dataIn.len / dimIn, dataOut.len / dimOut });
        return err.invalidNData;
    }

    // Create the header with standard values
    const header = DataHeader{
        .precision = @sizeOf(T),
        .nData = @as(u64, dataIn.len) / dimIn,
        .dimIn = dimIn,
        .dimOut = dimOut,
    };

    // Write the header to the file
    try file.writeStreamingAll(io, std.mem.asBytes(&header));

    // Write the data to the file
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(dataIn));
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(dataOut));
}

/// Loads the data points from a binary file. Allocates and returns the input and output
/// arrays; the caller is responsible for freeing both.
pub fn loadData(allocator: std.mem.Allocator, io: std.Io, fileName: []const u8) !struct { []T, []T } {
    // Open the file in fileName
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, fileName, .{});
    defer file.close(io);

    // Create the buffer to read the header
    var read_buffer: [4096]u8 = undefined;
    var file_reader = file.reader(io, &read_buffer);
    var reader = &file_reader.interface;

    // Read the header from the file
    var header: DataHeader = undefined;
    try reader.readSliceAll(std.mem.asBytes(&header));

    // Alloc the dataIn and dataOut arrays
    const dataIn = try allocator.alloc(T, header.nData * header.dimIn);
    const dataOut = try allocator.alloc(T, header.nData * header.dimOut);

    // Read the data from the file
    try reader.readSliceAll(std.mem.sliceAsBytes(dataIn));
    try reader.readSliceAll(std.mem.sliceAsBytes(dataOut));

    // Return the data
    return .{ dataIn, dataOut };
}

// Test the saving and loading of the data
test "[io] save/load-Data" {
    // Create the IO context
    const path = "test.bin";
    const io = std.testing.io;

    // Create the allocator
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) std.log.err("Leak!\n", .{});
    }

    // Set the random generator
    var xoshiro256 = std.Random.Xoshiro256.init(12345);
    const rand = std.Random.Xoshiro256.random(&xoshiro256);

    // Create some data points
    const nData: usize = 10;
    const nDim: usize = 4;
    var dataIn: [nData * nDim]T = undefined;
    var dataOut: [nData * nDim]T = undefined;
    for (0..nDim * nData) |i| {
        dataIn[i] = @floatCast(std.Random.floatNorm(rand, f32));
        dataOut[i] = @floatCast(std.Random.floatNorm(rand, f32));
    }

    // Save the data points to a file
    try saveData(io, path, &dataIn, &dataOut, nDim, nDim);

    // Load the weights from the same file
    const result = try loadData(allocator, io, path);
    defer allocator.free(result[0]);
    defer allocator.free(result[1]);

    // Check the data
    for (0..nDim) |i| {
        try testing.expectEqual(dataIn[i], result[0][i]);
        try testing.expectEqual(dataOut[i], result[1][i]);
    }

    // Delete the test file
    const cwd = std.Io.Dir.cwd();
    cwd.deleteFile(io, path) catch {};
}
