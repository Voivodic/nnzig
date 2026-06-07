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
const fType = params.floatType;

/// Header structure for neural network binary files, storing precision and layer topology
const NNHeader = struct {
    precision: u64 = @sizeOf(params.floatType),
    nLayers: u64 = @as(u64, params.nNeurons.len),
    nNeurons: [params.nNeurons.len]u64 = @as([params.nNeurons.len]u64, params.nNeurons),
};

/// Header structure for generic data binary files, storing precision, count, and dimension
const DataHeader = struct {
    precision: u64 = @sizeOf(params.floatType),
    nData: u64 = 0,
    dimData: u64 = 0,
};

/// Save the weights of the neural network to a binary file
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
    try file.writeStreamingAll(nn.ioContext, std.mem.sliceAsBytes(nn.weights));
    try file.writeStreamingAll(nn.ioContext, std.mem.sliceAsBytes(nn.biases));

}

/// Load the weights of the neural network from a binary file
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
    if (header.precision != @sizeOf(params.floatType)) {
        std.log.err("Precision mismatch! Expected {} bits, got {} bits!", .{ @sizeOf(params.floatType), header.precision });
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
    try reader.readSliceAll(std.mem.sliceAsBytes(nn.weights));
    try reader.readSliceAll(std.mem.sliceAsBytes(nn.biases));
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
    nnIn.weights[0] = 2.0;
    nnIn.biases[0] = 1.0;

    // Save the weights to a file
    try saveWeights(path, &nnIn);

    // Load the weights from the same file
    try loadWeights(path, &nnOut);

    // Check the weights
    try testing.expectEqual(nnIn.weights[0], nnOut.weights[0]);
    try testing.expectEqual(nnIn.biases[0], nnOut.biases[0]);

    // Delete the test file
    const cwd = std.Io.Dir.cwd();
    cwd.deleteFile(io, path) catch {};
}

/// Save the data points to a binary file
pub fn saveData(io: std.Io, fileName: []const u8, data: []const fType, dimData: u64) !void {
    // Open the file in fileName
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.createFile(io, fileName, .{});
    defer file.close(io);

    // Check the size of the data
    if (data.len % dimData != 0) {
        std.log.err("The size of the data ({}) is not a multiple of dimData ({})!\n", .{data.len, dimData});
        return err.invalidNData;
    }

    // Create the header with standard values
    const header = DataHeader{
        .precision = @sizeOf(fType),
        .nData = @as(u64, data.len) / dimData,
        .dimData = dimData,
    };

    // Write the header to the file
    try file.writeStreamingAll(io, std.mem.asBytes(&header));

    // Write the data to the file
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(data));
}

/// Load the data points from a binary file
pub fn loadData(io: std.Io, fileName: []const u8, data: []fType) !u64 {
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

    // Check the header values
    if (header.precision != @sizeOf(@TypeOf(data[0]))) {
        std.log.err("Precision mismatch! Expected {} bits, got {} bits!", .{ @sizeOf(@TypeOf(data[0])), header.precision });
        return err.precisionMismatch;
    }
    if (header.nData * header.dimData != data.len) {
        std.log.err("Number of data points * dimension mismatch! Expected {}, got {}!", .{ header.nData * header.dimData, data.len });
        return err.invalidNData;
    }

    // Read the data from the file
    try reader.readSliceAll(std.mem.sliceAsBytes(data));

    return header.dimData;
}

// Test the saving and loading of the data 
test "[io] save/load-Data" {
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

    // Set the random generator
    var xoshiro256 = std.Random.Xoshiro256.init(12345);
    const rand = std.Random.Xoshiro256.random(&xoshiro256);

    // Create some data points
    const nData: usize = 10;
    const nDim: usize = 10;
    var dataIn: []fType = undefined;
    var dataOut: []fType = undefined;
    dataIn = try allocator.alloc(fType, nData * nDim);
    dataOut = try allocator.alloc(fType, nData * nDim);
    defer {
        allocator.free(dataIn);
        allocator.free(dataOut);
    }
    for (dataIn) |*dIn| {
        dIn.* = std.Random.floatNorm(rand, fType);
    }

    // Save the data points to a file
    try saveData(io, path, dataIn, nDim);

    // Load the weights from the same file
    _ = try loadData(io, path, dataOut);

    // Check the data
    for (dataIn, dataOut) |*dIn, *dOut| {
        try testing.expectEqual(dIn.*, dOut.*);
    }

    // Delete the test file
    const cwd = std.Io.Dir.cwd();
    cwd.deleteFile(io, path) catch {};
}
