//! Zig wrappers around Eigen C++ linear algebra routines. Provides type-safe,
//! generic functions for matrix-vector operations used in neural network
//! forward and backward passes. Supports f16, f32, and f64 precision.
const testing = @import("std").testing;

/// Multiply a matrix and a vector then add another vector (M * a + b)
pub extern "c" fn eigenf16_matrixVectorMulAdd(a: [*]const f16, b: [*]const f16, c: [*]const f16, d: [*]f16, a_rows: usize, a_cols: usize) void;
pub extern "c" fn eigenf32_matrixVectorMulAdd(a: [*]const f32, b: [*]const f32, c: [*]const f32, d: [*]f32, a_rows: usize, a_cols: usize) void;
pub extern "c" fn eigenf64_matrixVectorMulAdd(a: [*]const f64, b: [*]const f64, c: [*]const f64, d: [*]f64, a_rows: usize, a_cols: usize) void;

/// Computes `result = matrix * vector_mult + vector_add` using Eigen
pub fn matrixVectorMulAdd(comptime T: type, matrix: []const T, vector_mult: []const T, vector_add: []const T, vector_result: []T) void {
    switch (T) {
        f16 => eigenf16_matrixVectorMulAdd(matrix.ptr, vector_mult.ptr, vector_add.ptr, vector_result.ptr, vector_add.len, vector_mult.len),
        f32 => eigenf32_matrixVectorMulAdd(matrix.ptr, vector_mult.ptr, vector_add.ptr, vector_result.ptr, vector_add.len, vector_mult.len),
        f64 => eigenf64_matrixVectorMulAdd(matrix.ptr, vector_mult.ptr, vector_add.ptr, vector_result.ptr, vector_add.len, vector_mult.len),
        else => @compileError("Unsupported type used in matrixVectorMulAdd!"),
    }
}

test "[eigen] matrixVectorMulAdd" {
    const testing_types = [_]type{ f16, f32, f64 };
    inline for (testing_types) |T| {
        const matrix = [_]T{ 1.0, 2.0, 3.0, 4.0 };
        const v_mult = [_]T{ 1.0, 2.0 };
        const v_add = [_]T{ 5.0, 6.0 };
        var result = [_]T{ 0.0, 0.0 };

        matrixVectorMulAdd(T, matrix[0..], v_mult[0..], v_add[0..], result[0..]);

        try testing.expectApproxEqAbs(12.0, result[0], 0.001);
        try testing.expectApproxEqAbs(16.0, result[1], 0.001);
    }
}

/// Multiply a vector by a matrix
pub extern "c" fn eigenf16_vectorMatrixMul(A: [*]f16, B: [*]const f16, b_rows: usize, b_cols: usize) void;
pub extern "c" fn eigenf32_vectorMatrixMul(A: [*]f32, B: [*]const f32, b_rows: usize, b_cols: usize) void;
pub extern "c" fn eigenf64_vectorMatrixMul(A: [*]f64, B: [*]const f64, b_rows: usize, b_cols: usize) void;

/// Computes `vector_mult *= matrix` (in-place vector-matrix multiplication) using Eigen
pub fn vectorMatrixMul(comptime T: type, vector_mult: []T, matrix: []const T) void {
    switch (T) {
        f16 => eigenf16_vectorMatrixMul(vector_mult.ptr, matrix.ptr, vector_mult.len, matrix.len / vector_mult.len),
        f32 => eigenf32_vectorMatrixMul(vector_mult.ptr, matrix.ptr, vector_mult.len, matrix.len / vector_mult.len),
        f64 => eigenf64_vectorMatrixMul(vector_mult.ptr, matrix.ptr, vector_mult.len, matrix.len / vector_mult.len),
        else => @compileError("Unsupported type used in vectorMatrixMul!"),
    }
}

test "[eigen] vectorMatrixMul" {
    const testing_types = [_]type{ f16, f32, f64 };
    inline for (testing_types) |T| {
        var v = [_]T{ 1.0, 2.0 };
        const matrix = [_]T{ 1.0, 2.0, 3.0, 4.0 };

        vectorMatrixMul(T, v[0..], matrix[0..]);

        try testing.expectApproxEqAbs(5.0, v[0], 0.001);
        try testing.expectApproxEqAbs(11.0, v[1], 0.001);
    }
}

/// Multiply two vectors elementwise
pub extern "c" fn eigenf16_vectorMul(A: [*]const f16, B: [*]f16, size: usize) void;
pub extern "c" fn eigenf32_vectorMul(A: [*]const f32, B: [*]f32, size: usize) void;
pub extern "c" fn eigenf64_vectorMul(A: [*]const f64, B: [*]f64, size: usize) void;

/// Computes `vector_b *= vector_a` element-wise using Eigen
pub fn vectorMul(comptime T: type, vector_a: []const T, vector_b: []T) void {
    switch (T) {
        f16 => eigenf16_vectorMul(vector_a.ptr, vector_b.ptr, vector_b.len),
        f32 => eigenf32_vectorMul(vector_a.ptr, vector_b.ptr, vector_b.len),
        f64 => eigenf64_vectorMul(vector_a.ptr, vector_b.ptr, vector_b.len),
        else => @compileError("Unsupported type used in eigen_vectorMul!"),
    }
}

test "[eigen] vectorMul" {
    const testing_types = [_]type{ f16, f32, f64 };
    inline for (testing_types) |T| {
        const a = [_]T{ 2.0, 3.0 };
        var b = [_]T{ 4.0, 5.0 };

        vectorMul(T, a[0..], b[0..]);

        try testing.expectApproxEqAbs(8.0, b[0], 0.001);
        try testing.expectApproxEqAbs(15.0, b[1], 0.001);
    }
}

/// Set an array to zero
pub extern "c" fn eigenf16_setZero(A: [*]f16, size: usize) void;
pub extern "c" fn eigenf32_setZero(A: [*]f32, size: usize) void;
pub extern "c" fn eigenf64_setZero(A: [*]f64, size: usize) void;

/// Sets all elements of the vector to zero using Eigen
pub fn setZero(comptime T: type, vector: []T) void {
    switch (T) {
        f16 => eigenf16_setZero(vector.ptr, vector.len),
        f32 => eigenf32_setZero(vector.ptr, vector.len),
        f64 => eigenf64_setZero(vector.ptr, vector.len),
        else => @compileError("Unsupported type used in eigen_setZero!"),
    }
}

test "[eigen] setZero" {
    const testing_types = [_]type{ f16, f32, f64 };
    inline for (testing_types) |T| {
        var v = [_]T{ 1.0, 2.0, 3.0 };

        setZero(T, v[0..]);

        try testing.expectApproxEqAbs(0.0, v[0], 0.001);
        try testing.expectApproxEqAbs(0.0, v[1], 0.001);
        try testing.expectApproxEqAbs(0.0, v[2], 0.001);
    }
}

/// Initialize a vector with the values from another vector
pub extern "c" fn eigenf16_vectorInit(A: [*]const f16, B: [*]f16, size: usize) void;
pub extern "c" fn eigenf32_vectorInit(A: [*]const f32, B: [*]f32, size: usize) void;
pub extern "c" fn eigenf64_vectorInit(A: [*]const f64, B: [*]f64, size: usize) void;

/// Copies all elements from `vector_a` into `vector_b` using Eigen
pub fn vectorInit(comptime T: type, vector_a: []const T, vector_b: []T) void {
    switch (T) {
        f16 => eigenf16_vectorInit(vector_a.ptr, vector_b.ptr, vector_b.len),
        f32 => eigenf32_vectorInit(vector_a.ptr, vector_b.ptr, vector_b.len),
        f64 => eigenf64_vectorInit(vector_a.ptr, vector_b.ptr, vector_b.len),
        else => @compileError("Unsupported type used in eigen_vectorInit!"),
    }
}

test "[eigen] vectorInit" {
    const testing_types = [_]type{ f16, f32, f64 };
    inline for (testing_types) |T| {
        const a = [_]T{ 1.0, 2.0, 3.0 };
        var b = [_]T{ 0.0, 0.0, 0.0 };

        vectorInit(T, a[0..], b[0..]);

        try testing.expectApproxEqAbs(1.0, b[0], 0.001);
        try testing.expectApproxEqAbs(2.0, b[1], 0.001);
        try testing.expectApproxEqAbs(3.0, b[2], 0.001);
    }
}

/// Accumulates the outer product of `v` and `y` into the weight gradient
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

test "[eigen] updateGradWeights" {
    const testing_types = [_]type{ f16, f32, f64 };
    inline for (testing_types) |T| {
        const v = [_]T{ 1.0, 2.0 };
        const y = [_]T{ 0.5, 0.5 };
        var grad = [_]T{ 0.0, 0.0, 0.0, 0.0 };

        updateGradWeights(T, v[0..], y[0..], grad[0..]);

        try testing.expectApproxEqAbs(0.5, grad[0], 0.001);
        try testing.expectApproxEqAbs(1.0, grad[1], 0.001);
        try testing.expectApproxEqAbs(0.5, grad[2], 0.001);
        try testing.expectApproxEqAbs(1.0, grad[3], 0.001);
    }
}

/// Update mt and vt for the biases
pub extern "c" fn eigenf16_updateGradBiases(V: [*]const f16, grad: [*]f16, v_size: usize) void;
pub extern "c" fn eigenf32_updateGradBiases(V: [*]const f32, grad: [*]f32, v_size: usize) void;
pub extern "c" fn eigenf64_updateGradBiases(V: [*]const f64, grad: [*]f64, v_size: usize) void;

/// Adds `v` element-wise into the bias gradient
pub fn updateGradBiases(comptime T: type, v: []const T, grad: []T) void {
    switch (T) {
        f16 => eigenf16_updateGradBiases(v.ptr, grad.ptr, v.len),
        f32 => eigenf32_updateGradBiases(v.ptr, grad.ptr, v.len),
        f64 => eigenf64_updateGradBiases(v.ptr, grad.ptr, v.len),
        else => @compileError("Unsupported type used in eigen_updateGradBiases!"),
    }
}

test "[eigen] updateGradBiases" {
    const testing_types = [_]type{ f16, f32, f64 };
    inline for (testing_types) |T| {
        const v = [_]T{ 1.0, 2.0 };
        var grad = [_]T{ 1.0, 2.0 };

        updateGradBiases(T, v[0..], grad[0..]);

        try testing.expectApproxEqAbs(2.0, grad[0], 0.001);
        try testing.expectApproxEqAbs(4.0, grad[1], 0.001);
    }
}
