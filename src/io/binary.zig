//! Implementation of functions to read and write the weights of the
//! neural network to and from a safetensors files.

// Import the modules
const std = @import("std");
const nnzig = @import("nnzig");
const params = @import("params");
const errors = @import("errors");
const err = errors.ioError;
const testing = std.testing;

// Header for the neural network binary files
const NNHeader = struct {
    precision: u64 = @sizeOf(params.floatType),
    nLayers: u64 = @as(u64, params.nNeurons.len),
    nNeurons: [params.nNeurons.len]u64 = @as([params.nNeurons.len]u64, params.nNeurons),
};

// Header for the data binary files
const DataHeader = struct {
    precision: u64 = @sizeOf(params.floatType),
    nData: u64 = 0,
    dimData: u64 = 0,
};

/// Save the weights of the neural network to a binary file
pub fn saveWeights(comptime T: type, io: std.Io, fileName: []const u8, nn: *const nnzig.NN(T)) !void {
    // Open the file in fileName
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.createFile(io, fileName, .{});
    defer file.close(io);

    // Create the header with standard values
    const header = NNHeader{};

    // Write the header to the file
    try file.writeStreamingAll(io, std.mem.asBytes(&header));

    // Write the weights and biases to the file
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(nn.weights));
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(nn.biases));

}

/// Load the weights of the neural network from a binary file
pub fn loadWeights(comptime T: type, io: std.Io, fileName: []const u8, nn: *nnzig.NN(T)) !void {
    // Open the file in fileName
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, fileName, .{});
    defer file.close(io);

    // Create the buffer to read the header
    var read_buffer: [4096]u8 = undefined;
    var file_reader = file.reader(io, &read_buffer);
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
    var nnIn = try nnzig.NN(params.floatType).init(allocator);
    defer nnIn.deinit();
    var nnOut = try nnzig.NN(params.floatType).init(allocator);
    defer nnOut.deinit();

    // Change some weights
    nnIn.weights[0] = 2.0;
    nnIn.biases[0] = 1.0;

    // Save the weights to a file
    try saveWeights(params.floatType, io, path, &nnIn);

    // Load the weights from the same file
    try loadWeights(params.floatType, io, path, &nnOut);

    // Check the weights
    try testing.expectEqual(nnIn.weights[0], nnOut.weights[0]);
    try testing.expectEqual(nnIn.biases[0], nnOut.biases[0]);

    // Delete the test file
    const cwd = std.Io.Dir.cwd();
    cwd.deleteFile(io, path) catch {};
}

/// Save the data points to a binary file
pub fn saveData(comptime T: type, io: std.Io, fileName: []const u8, data: []const T, dimData: u64) !void {
    // Open the file in fileName
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.createFile(io, fileName, .{});
    defer file.close(io);

    // Create the header with standard values
    const header = DataHeader{
        .precision = @sizeOf(T),
        .nData = @as(u64, data.len) / dimData,
        .dimData = dimData,
    };

    // Write the header to the file
    try file.writeStreamingAll(io, std.mem.asBytes(&header));

    // Write the data to the file
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(data));
}

/// Load the data points from a binary file
pub fn loadData(comptime T: type, io: std.Io, fileName: []const u8, data: []T) !u64 {
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
    var dataIn: []f32 = undefined;
    var dataOut: []f32 = undefined;
    dataIn = try allocator.alloc(f32, nData * nDim);
    dataOut = try allocator.alloc(f32, nData * nDim);
    defer {
        allocator.free(dataIn);
        allocator.free(dataOut);
    }
    for (dataIn) |*dIn| {
        dIn.* = std.Random.floatNorm(rand, f32);
    }

    // Save the data points to a file
    try saveData(params.floatType, io, path, dataIn, nDim);

    // Load the weights from the same file
    _ = try loadData(params.floatType, io, path, dataOut);

    // Check the data
    for (dataIn, dataOut) |*dIn, *dOut| {
        try testing.expectEqual(dIn.*, dOut.*);
    }

    // Delete the test file
    const cwd = std.Io.Dir.cwd();
    cwd.deleteFile(io, path) catch {};
}
