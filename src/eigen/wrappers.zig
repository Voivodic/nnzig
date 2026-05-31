//! Wrappers for the Eigen library. It provides a safe interface for calling Eigen functions from Zig.

/// Multiply a matrix and a vector then add another vector
pub extern "c" fn eigenf16_matrixVectorMulAdd(a: [*]const f16, b: [*]const f16, c: [*]const f16, d: [*]f16, a_rows: usize, a_cols: usize) void;
pub extern "c" fn eigenf32_matrixVectorMulAdd(a: [*]const f32, b: [*]const f32, c: [*]const f32, d: [*]f32, a_rows: usize, a_cols: usize) void;
pub extern "c" fn eigenf64_matrixVectorMulAdd(a: [*]const f64, b: [*]const f64, c: [*]const f64, d: [*]f64, a_rows: usize, a_cols: usize) void;

pub fn matrixVectorMulAdd(comptime T: type, matrix: []const T, vector_mult: []const T, vector_add: []const T, vector_result: []T) void {
    switch (T) {
        f16 => eigenf16_matrixVectorMulAdd(matrix.ptr, vector_mult.ptr, vector_add.ptr, vector_result.ptr, vector_add.len, vector_mult.len),
        f32 => eigenf32_matrixVectorMulAdd(matrix.ptr, vector_mult.ptr, vector_add.ptr, vector_result.ptr, vector_add.len, vector_mult.len),
        f64 => eigenf64_matrixVectorMulAdd(matrix.ptr, vector_mult.ptr, vector_add.ptr, vector_result.ptr, vector_add.len, vector_mult.len),
        else => @compileError("Unsupported type used in matrixVectorMulAdd!"),
    }
}

/// Multiply a vector by a matrix
pub extern "c" fn eigenf16_vectorMatrixMul(A: [*]f16, B: [*]const f16, b_rows: usize, b_cols: usize) void;
pub extern "c" fn eigenf32_vectorMatrixMul(A: [*]f32, B: [*]const f32, b_rows: usize, b_cols: usize) void;
pub extern "c" fn eigenf64_vectorMatrixMul(A: [*]f64, B: [*]const f64, b_rows: usize, b_cols: usize) void;

pub fn vectorMatrixMul(comptime T: type, vector_mult: []T, matrix: []const T) void {
    switch (T) {
        f16 => eigenf16_vectorMatrixMul(vector_mult.ptr, matrix.ptr, vector_mult.len, matrix.len / vector_mult.len),
        f32 => eigenf32_vectorMatrixMul(vector_mult.ptr, matrix.ptr, vector_mult.len, matrix.len / vector_mult.len),
        f64 => eigenf64_vectorMatrixMul(vector_mult.ptr, matrix.ptr, vector_mult.len, matrix.len / vector_mult.len),
        else => @compileError("Unsupported type used in vectorMatrixMul!"),
    }
}

/// Multiply two vectors elementwise
pub extern "c" fn eigenf16_vectorMul(A: [*]const f16, B: [*]f16, size: usize) void;
pub extern "c" fn eigenf32_vectorMul(A: [*]const f32, B: [*]f32, size: usize) void;
pub extern "c" fn eigenf64_vectorMul(A: [*]const f64, B: [*]f64, size: usize) void;

pub fn vectorMul(comptime T: type, vector_a: []const T, vector_b: []T) void {
    switch (T) {
        f16 => eigenf16_vectorMul(vector_a.ptr, vector_b.ptr, vector_b.len),
        f32 => eigenf32_vectorMul(vector_a.ptr, vector_b.ptr, vector_b.len),
        f64 => eigenf64_vectorMul(vector_a.ptr, vector_b.ptr, vector_b.len),
        else => @compileError("Unsupported type used in eigen_vectorMul!"),
    }
}

/// Set an array to zero
pub extern "c" fn eigenf16_setZero(A: [*]f16, size: usize) void;
pub extern "c" fn eigenf32_setZero(A: [*]f32, size: usize) void;
pub extern "c" fn eigenf64_setZero(A: [*]f64, size: usize) void;

pub fn setZero(comptime T: type, vector: []T) void {
    switch (T) {
        f16 => eigenf16_setZero(vector.ptr, vector.len),
        f32 => eigenf32_setZero(vector.ptr, vector.len),
        f64 => eigenf64_setZero(vector.ptr, vector.len),
        else => @compileError("Unsupported type used in eigen_setZero!"),
    }
}

/// Initialize a vector with the values from another vector
pub extern "c" fn eigenf16_vectorInit(A: [*]const f16, B: [*]f16, size: usize) void;
pub extern "c" fn eigenf32_vectorInit(A: [*]const f32, B: [*]f32, size: usize) void;
pub extern "c" fn eigenf64_vectorInit(A: [*]const f64, B: [*]f64, size: usize) void;

pub fn vectorInit(comptime T: type, vector_a: []const T, vector_b: []T) void {
    switch (T) {
        f16 => eigenf16_vectorInit(vector_a.ptr, vector_b.ptr, vector_b.len),
        f32 => eigenf32_vectorInit(vector_a.ptr, vector_b.ptr, vector_b.len),
        f64 => eigenf64_vectorInit(vector_a.ptr, vector_b.ptr, vector_b.len),
        else => @compileError("Unsupported type used in eigen_vectorInit!"),
    }
}

// Update mt and vt for the weights
pub extern "c" fn eigenf16_updateGradWeights(V: [*]const f16, Y: [*]const f16, grad: [*]f16, v_size: usize, y_size: usize) void;
pub extern "c" fn eigenf32_updateGradWeights(V: [*]const f32, Y: [*]const f32, grad: [*]f32, v_size: usize, y_size: usize) void;
pub extern "c" fn eigenf64_updateGradWeights(V: [*]const f64, Y: [*]const f64, grad: [*]f64, v_size: usize, y_size: usize) void;

pub fn updateGradWeights(comptime T: type, v: []const T, y: []const T, grad: []T) void {
    switch (T) {
        f16 => eigenf16_updateGradWeights(v.ptr, y.ptr, grad.ptr, v.len, y.len),
        f32 => eigenf32_updateGradWeights(v.ptr, y.ptr, grad.ptr, v.len, y.len),
        f64 => eigenf64_updateGradWeights(v.ptr, y.ptr, grad.ptr, v.len, y.len),
        else => @compileError("Unsupported type used in eigen_updateGradWeights!"),
    }
}

/// Update mt and vt for the biases
pub extern "c" fn eigenf16_updateGradBiases(V: [*]const f16, grad: [*]f16, v_size: usize) void;
pub extern "c" fn eigenf32_updateGradBiases(V: [*]const f32, grad: [*]f32, v_size: usize) void;
pub extern "c" fn eigenf64_updateGradBiases(V: [*]const f64, grad: [*]f64, v_size: usize) void;

pub fn updateGradBiases(comptime T: type, v: []const T, grad: []T) void {
    switch (T) {
        f16 => eigenf16_updateGradBiases(v.ptr, grad.ptr, v.len),
        f32 => eigenf32_updateGradBiases(v.ptr, grad.ptr, v.len),
        f64 => eigenf64_updateGradBiases(v.ptr, grad.ptr, v.len),
        else => @compileError("Unsupported type used in eigen_updateGradBiases!"),
    }
}
