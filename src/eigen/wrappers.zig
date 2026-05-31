//! Eigen wrappers

/// Multiply a matrix and a vector then add another vector
pub extern "c" fn eigen_matrixVectorMulAdd(a: [*]const f32, b: [*]const f32, c: [*]const f32, d: [*]f32, a_rows: usize, a_cols: usize) void;

pub fn matrixVectorMulAdd(comptime T: type, matrix: []const f32, vector_mult: []const f32, vector_add: []const f32, vector_result: []f32) void {
    switch (T) {
        f32 => eigen_matrixVectorMulAdd(matrix.ptr, vector_mult.ptr, vector_add.ptr, vector_result.ptr, vector_add.len, vector_mult.len),
        else => @compileError("Unsupported type used in matrixVectorMulAdd!"),
    }
}

/// Multiply a vector by a matrix
pub extern "c" fn eigen_vectorMatrixMul(A: [*]f32, B: [*]const f32, b_rows: usize, b_cols: usize) void;

pub fn vectorMatrixMul(comptime T: type, vector_mult: []f32, matrix: []const f32) void {
    switch (T) {
        f32 => eigen_vectorMatrixMul(vector_mult.ptr, matrix.ptr, vector_mult.len, matrix.len / vector_mult.len),
        else => @compileError("Unsupported type used in vectorMatrixMul!"),
    }
}

/// Multiply two vectors elementwise
pub extern "c" fn eigen_vectorMul(A: [*]const f32, B: [*]f32, size: usize) void;

pub fn vectorMul(T: type, vector_a: []const f32, vector_b: []f32) void {
    switch (T) {
        f32 => eigen_vectorMul(vector_a.ptr, vector_b.ptr, vector_b.len),
        else => @compileError("Unsupported type used in eigen_vectorMul!"),
    }
}

/// Set an array to zero
pub extern "c" fn eigen_setZero(A: [*]f32, size: usize) void;

pub fn setZero(T: type, vector: []f32) void {
    switch (T) {
        f32 => eigen_setZero(vector.ptr, vector.len),
        else => @compileError("Unsupported type used in eigen_setZero!"),
    }
}

/// Initialize a vector with the values from another vector
pub extern "c" fn eigen_vectorInit(A: [*]const f32, B: [*]f32, size: usize) void;

pub fn vectorInit(T: type, vector_a: []const f32, vector_b: []f32) void {
    switch (T) {
        f32 => eigen_vectorInit(vector_a.ptr, vector_b.ptr, vector_b.len),
        else => @compileError("Unsupported type used in eigen_vectorInit!"),
    }
}

// Update mt and vt for the weights
pub extern "c" fn eigen_updateGradWeights(V: [*]const f32, Y: [*]const f32, grad: [*]f32, v_size: usize, y_size: usize) void;

pub fn updateGradWeights(T: type, v: []const f32, y: []const f32, grad: []f32) void {
    switch (T) {
        f32 => eigen_updateGradWeights(v.ptr, y.ptr, grad.ptr, v.len, y.len),
        else => @compileError("Unsupported type used in eigen_updateGradWeights!"),
    }
}

/// Update mt and vt for the biases
pub extern "c" fn eigen_updateGradBiases(V: [*]const f32, grad: [*]f32, v_size: usize) void;

pub fn updateGradBiases(T: type, v: []const f32, grad: []f32) void {
    switch (T) {
        f32 => eigen_updateGradBiases(v.ptr, grad.ptr, v.len),
        else => @compileError("Unsupported type used in eigen_updateGradBiases!"),
    }
}
