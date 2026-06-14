//! Zig wrappers around Eigen C++ linear algebra routines. Provides functions
//! for matrix-vector operations used in neural network forward and backward
//! passes. The float precision is selected at compile time via params.T.
const testing = @import("std").testing;
const params = @import("params");

const T = params.T;

// --- Matrix * vector + vector ---

extern "c" fn eigen_matrixVectorMulAdd(a: [*]const T, b: [*]const T, c: [*]const T, d: [*]T, a_rows: usize, a_cols: usize) void;

/// Computes `result = matrix * vector_mult + vector_add` using Eigen
pub fn matrixVectorMulAdd(matrix: []const T, vector_mult: []const T, vector_add: []const T, vector_result: []T) void {
    eigen_matrixVectorMulAdd(matrix.ptr, vector_mult.ptr, vector_add.ptr, vector_result.ptr, vector_add.len, vector_mult.len);
}

test "[eigen] matrixVectorMulAdd" {
    const matrix = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    const v_mult = [_]T{ 1.0, 2.0 };
    const v_add = [_]T{ 5.0, 6.0 };
    var result = [_]T{ 0.0, 0.0 };

    matrixVectorMulAdd(matrix[0..], v_mult[0..], v_add[0..], result[0..]);

    try testing.expectApproxEqAbs(12.0, result[0], 0.001);
    try testing.expectApproxEqAbs(16.0, result[1], 0.001);
}

extern "c" fn eigen_matrixVectorMulAddBatch(a: [*]const T, b: [*]const T, c: [*]const T, d: [*]T, a_rows: usize, a_cols: usize, batch_size: usize) void;

/// Computes `result = matrix * vector_mult + vector_add` using Eigen for a full batch
pub fn matrixVectorMulAddBatch(matrix: []const T, vector_mult: []const T, vector_add: []const T, vector_result: []T) void {
    const dimOut: usize = vector_add.len;
    const dimIn: usize = matrix.len / dimOut;
    const batchSize: usize = vector_mult.len / dimIn;

    eigen_matrixVectorMulAddBatch(matrix.ptr, vector_mult.ptr, vector_add.ptr, vector_result.ptr, dimOut, dimIn, batchSize);
}

test "[eigen] matrixVectorMulAddBatch" {
    const matrix = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    const v_mult = [_]T{ 1.0, 2.0, 1.0, 2.0 };
    const v_add = [_]T{ 5.0, 6.0 };
    var result = [_]T{ 0.0, 0.0, 0.0, 0.0 };

    matrixVectorMulAddBatch(matrix[0..], v_mult[0..], v_add[0..], result[0..]);

    try testing.expectApproxEqAbs(12.0, result[0], 0.001);
    try testing.expectApproxEqAbs(16.0, result[1], 0.001);
    try testing.expectApproxEqAbs(12.0, result[2], 0.001);
    try testing.expectApproxEqAbs(16.0, result[3], 0.001);
}

// --- Vector * matrix ---

extern "c" fn eigen_vectorMatrixMul(A: [*]T, B: [*]const T, b_rows: usize, b_cols: usize) void;

/// Computes `vector_mult *= matrix` (in-place vector-matrix multiplication) using Eigen
pub fn vectorMatrixMul(vector_mult: []T, matrix: []const T) void {
    eigen_vectorMatrixMul(vector_mult.ptr, matrix.ptr, vector_mult.len, matrix.len / vector_mult.len);
}

test "[eigen] vectorMatrixMul" {
    var v = [_]T{ 1.0, 2.0 };
    const matrix = [_]T{ 1.0, 2.0, 3.0, 4.0 };

    vectorMatrixMul(v[0..], matrix[0..]);

    try testing.expectApproxEqAbs(5.0, v[0], 0.001);
    try testing.expectApproxEqAbs(11.0, v[1], 0.001);
}

extern "c" fn eigen_vectorMatrixMulBatch(A: [*]const T, result: [*]T, B: [*]const T, b_rows: usize, b_cols: usize, batch_size: usize) void;

/// Computes `result = vectors^T * matrix` for a batch of row vectors using Eigen
pub fn vectorMatrixMulBatch(vectors_in: []const T, vectors_out: []T, matrix: []const T, in_dim: usize) void {
    const batch_size: usize = vectors_in.len / in_dim;
    const out_dim: usize = matrix.len / in_dim;

    eigen_vectorMatrixMulBatch(vectors_in.ptr, vectors_out.ptr, matrix.ptr, in_dim, out_dim, batch_size);
}

test "[eigen] vectorMatrixMulBatch" {
    const vectors_in = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    const matrix = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    var vectors_out = [_]T{ 0.0, 0.0, 0.0, 0.0 };

    vectorMatrixMulBatch(vectors_in[0..], vectors_out[0..], matrix[0..], 2);

    try testing.expectApproxEqAbs(5.0, vectors_out[0], 0.001);
    try testing.expectApproxEqAbs(11.0, vectors_out[1], 0.001);
    try testing.expectApproxEqAbs(11.0, vectors_out[2], 0.001);
    try testing.expectApproxEqAbs(25.0, vectors_out[3], 0.001);
}

// --- Vector * vector (elementwise) ---

extern "c" fn eigen_vectorMul(A: [*]const T, B: [*]T, size: usize) void;

/// Computes `vector_b *= vector_a` element-wise using Eigen
pub fn vectorMul(vector_a: []const T, vector_b: []T) void {
    eigen_vectorMul(vector_a.ptr, vector_b.ptr, vector_b.len);
}

test "[eigen] vectorMul" {
    const a = [_]T{ 2.0, 3.0 };
    var b = [_]T{ 4.0, 5.0 };

    vectorMul(a[0..], b[0..]);

    try testing.expectApproxEqAbs(8.0, b[0], 0.001);
    try testing.expectApproxEqAbs(15.0, b[1], 0.001);
}

// --- Vector utilities ---

extern "c" fn eigen_setZero(A: [*]T, size: usize) void;

/// Sets all elements of the vector to zero using Eigen
pub fn setZero(vector: []T) void {
    eigen_setZero(vector.ptr, vector.len);
}

test "[eigen] setZero" {
    var v = [_]T{ 1.0, 2.0, 3.0 };

    setZero(v[0..]);

    try testing.expectApproxEqAbs(0.0, v[0], 0.001);
    try testing.expectApproxEqAbs(0.0, v[1], 0.001);
    try testing.expectApproxEqAbs(0.0, v[2], 0.001);
}

extern "c" fn eigen_vectorInit(A: [*]const T, B: [*]T, size: usize) void;

/// Copies all elements from `vector_a` into `vector_b` using Eigen
pub fn vectorInit(vector_a: []const T, vector_b: []T) void {
    eigen_vectorInit(vector_a.ptr, vector_b.ptr, vector_b.len);
}

test "[eigen] vectorInit" {
    const a = [_]T{ 1.0, 2.0, 3.0 };
    var b = [_]T{ 0.0, 0.0, 0.0 };

    vectorInit(a[0..], b[0..]);

    try testing.expectApproxEqAbs(1.0, b[0], 0.001);
    try testing.expectApproxEqAbs(2.0, b[1], 0.001);
    try testing.expectApproxEqAbs(3.0, b[2], 0.001);
}

// --- Gradient accumulation (weights) ---

extern "c" fn eigen_updateGradWeights(V: [*]const T, Y: [*]const T, grad: [*]T, v_size: usize, y_size: usize) void;

/// Accumulates the outer product of `v` and `y` into the weight gradient
pub fn updateGradWeights(v: []const T, y: []const T, grad: []T) void {
    eigen_updateGradWeights(v.ptr, y.ptr, grad.ptr, v.len, y.len);
}

test "[eigen] updateGradWeights" {
    const v = [_]T{ 1.0, 2.0 };
    const y = [_]T{ 0.5, 0.5 };
    var grad = [_]T{ 0.0, 0.0, 0.0, 0.0 };

    updateGradWeights(v[0..], y[0..], grad[0..]);

    try testing.expectApproxEqAbs(0.5, grad[0], 0.001);
    try testing.expectApproxEqAbs(1.0, grad[1], 0.001);
    try testing.expectApproxEqAbs(0.5, grad[2], 0.001);
    try testing.expectApproxEqAbs(1.0, grad[3], 0.001);
}

extern "c" fn eigen_updateGradWeightsBatch(V: [*]const T, Y: [*]const T, grad: [*]T, v_size: usize, y_size: usize, batch_size: usize) void;

/// Accumulates the outer products from a batch into the weight gradient (`grad += V * Y^T`)
pub fn updateGradWeightsBatch(v: []const T, y: []const T, grad: []T, v_size: usize) void {
    const y_size: usize = grad.len / v_size;
    const batch_size: usize = v.len / v_size;

    eigen_updateGradWeightsBatch(v.ptr, y.ptr, grad.ptr, v_size, y_size, batch_size);
}

test "[eigen] updateGradWeightsBatch" {
    const v = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    const y = [_]T{ 0.5, 0.5, 1.0, 1.0 };
    var grad = [_]T{ 0.0, 0.0, 0.0, 0.0 };

    updateGradWeightsBatch(v[0..], y[0..], grad[0..], 2);

    try testing.expectApproxEqAbs(3.5, grad[0], 0.001);
    try testing.expectApproxEqAbs(5.0, grad[1], 0.001);
    try testing.expectApproxEqAbs(3.5, grad[2], 0.001);
    try testing.expectApproxEqAbs(5.0, grad[3], 0.001);
}

// --- Gradient accumulation (biases) ---

extern "c" fn eigen_updateGradBiases(V: [*]const T, grad: [*]T, v_size: usize) void;

/// Adds `v` element-wise into the bias gradient
pub fn updateGradBiases(v: []const T, grad: []T) void {
    eigen_updateGradBiases(v.ptr, grad.ptr, v.len);
}

test "[eigen] updateGradBiases" {
    const v = [_]T{ 1.0, 2.0 };
    var grad = [_]T{ 1.0, 2.0 };

    updateGradBiases(v[0..], grad[0..]);

    try testing.expectApproxEqAbs(2.0, grad[0], 0.001);
    try testing.expectApproxEqAbs(4.0, grad[1], 0.001);
}

extern "c" fn eigen_updateGradBiasesBatch(V: [*]const T, grad: [*]T, v_size: usize, batch_size: usize) void;

/// Accumulates bias gradients from a batch (`grad += rowwise_sum(V)`)
pub fn updateGradBiasesBatch(v: []const T, grad: []T) void {
    const v_size: usize = grad.len;
    const batch_size: usize = v.len / v_size;

    eigen_updateGradBiasesBatch(v.ptr, grad.ptr, v_size, batch_size);
}

test "[eigen] updateGradBiasesBatch" {
    const v = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    var grad = [_]T{ 1.0, 1.0 };

    updateGradBiasesBatch(v[0..], grad[0..]);

    try testing.expectApproxEqAbs(5.0, grad[0], 0.001);
    try testing.expectApproxEqAbs(7.0, grad[1], 0.001);
}
