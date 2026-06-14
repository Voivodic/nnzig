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

// --- Activation functions ---

extern "c" fn eigen_none(input: [*]T, df: [*]T, n: usize) void;
extern "c" fn eigen_relu(input: [*]T, df: [*]T, n: usize) void;
extern "c" fn eigen_tanh(input: [*]T, df: [*]T, n: usize) void;
extern "c" fn eigen_sigmoid(input: [*]T, df: [*]T, n: usize) void;

/// Applies the "none" activation: leaves `input` untouched and sets every derivative in `df` to 1
pub fn activateNone(input: []T, df: []T) void {
    eigen_none(input.ptr, df.ptr, input.len);
}

test "[eigen] activateNone" {
    var input = [_]T{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var df: [5]T = undefined;

    activateNone(&input, &df);

    try testing.expectApproxEqAbs(-2.0, input[0], 0.001);
    try testing.expectApproxEqAbs(2.0, input[4], 0.001);
    for (df) |val| try testing.expectApproxEqAbs(1.0, val, 0.001);
}

/// Applies ReLU element-wise to `input` (in place) and writes its derivative into `df`
pub fn activateRelu(input: []T, df: []T) void {
    eigen_relu(input.ptr, df.ptr, input.len);
}

test "[eigen] activateRelu" {
    var input = [_]T{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var df: [5]T = undefined;

    activateRelu(&input, &df);

    try testing.expectApproxEqAbs(0.0, input[0], 0.001);
    try testing.expectApproxEqAbs(0.0, input[1], 0.001);
    try testing.expectApproxEqAbs(0.0, input[2], 0.001);
    try testing.expectApproxEqAbs(1.0, input[3], 0.001);
    try testing.expectApproxEqAbs(2.0, input[4], 0.001);

    try testing.expectApproxEqAbs(0.0, df[0], 0.001);
    try testing.expectApproxEqAbs(0.0, df[1], 0.001);
    try testing.expectApproxEqAbs(1.0, df[2], 0.001);
    try testing.expectApproxEqAbs(1.0, df[3], 0.001);
    try testing.expectApproxEqAbs(1.0, df[4], 0.001);
}

/// Applies tanh element-wise to `input` (in place) and writes its derivative into `df`
pub fn activateTanh(input: []T, df: []T) void {
    eigen_tanh(input.ptr, df.ptr, input.len);
}

test "[eigen] activateTanh" {
    var input = [_]T{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var df: [5]T = undefined;

    activateTanh(&input, &df);

    try testing.expectApproxEqAbs(-0.964028, input[0], 1e-4);
    try testing.expectApproxEqAbs(-0.761594, input[1], 1e-4);
    try testing.expectApproxEqAbs(0.0, input[2], 1e-4);
    try testing.expectApproxEqAbs(0.761594, input[3], 1e-4);
    try testing.expectApproxEqAbs(0.964028, input[4], 1e-4);

    try testing.expectApproxEqAbs(0.0706508, df[0], 1e-4);
    try testing.expectApproxEqAbs(0.419974, df[1], 1e-4);
    try testing.expectApproxEqAbs(1.0, df[2], 1e-4);
    try testing.expectApproxEqAbs(0.419974, df[3], 1e-4);
    try testing.expectApproxEqAbs(0.0706508, df[4], 1e-4);
}

/// Applies sigmoid element-wise to `input` (in place) and writes its derivative into `df`
pub fn activateSigmoid(input: []T, df: []T) void {
    eigen_sigmoid(input.ptr, df.ptr, input.len);
}

test "[eigen] activateSigmoid" {
    var input = [_]T{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var df: [5]T = undefined;

    activateSigmoid(&input, &df);

    try testing.expectApproxEqAbs(0.119203, input[0], 1e-4);
    try testing.expectApproxEqAbs(0.268941, input[1], 1e-4);
    try testing.expectApproxEqAbs(0.5, input[2], 1e-4);
    try testing.expectApproxEqAbs(0.731059, input[3], 1e-4);
    try testing.expectApproxEqAbs(0.880797, input[4], 1e-4);

    try testing.expectApproxEqAbs(0.104994, df[0], 1e-4);
    try testing.expectApproxEqAbs(0.196612, df[1], 1e-4);
    try testing.expectApproxEqAbs(0.25, df[2], 1e-4);
    try testing.expectApproxEqAbs(0.196612, df[3], 1e-4);
    try testing.expectApproxEqAbs(0.104994, df[4], 1e-4);
}

// --- Loss functions ---

extern "c" fn eigen_mse(pred: [*]const T, out: [*]const T, dL: [*]T, loss: *T, n: usize) void;

/// Computes the mean squared error `L = 0.5 * mean((pred - out)^2)` and writes the
/// derivative `dL = (pred - out) / n` using Eigen. Returns the scalar loss.
pub fn computeMSE(pred: []const T, out: []const T, dL: []T) T {
    var loss: T = 0.0;
    eigen_mse(pred.ptr, out.ptr, dL.ptr, &loss, pred.len);
    return loss;
}

test "[eigen] computeMSE" {
    // Known error: pred=3, out=1, n=4 -> diff=2, loss=0.5*mean(4)=2.0, dL=2/4=0.5
    const pred = [_]T{ 3.0, 3.0, 3.0, 3.0 };
    const out = [_]T{ 1.0, 1.0, 1.0, 1.0 };
    var dL: [4]T = undefined;

    const lossVal = computeMSE(&pred, &out, &dL);

    try testing.expectApproxEqAbs(2.0, lossVal, 1e-4);
    for (dL) |val| try testing.expectApproxEqAbs(0.5, val, 1e-4);

    // Zero error: pred == out -> loss=0, dL=0
    const lossZero = computeMSE(&pred, &pred, &dL);

    try testing.expectApproxEqAbs(0.0, lossZero, 1e-6);
    for (dL) |val| try testing.expectApproxEqAbs(0.0, val, 1e-6);
}

// --- Normalization functions ---

extern "c" fn eigen_computeMeanStd(inputs: [*]const T, outputs: [*]const T, aIn: [*]T, bIn: [*]T, aOut: [*]T, bOut: [*]T, nIn: usize, nOut: usize, nData: usize) void;
extern "c" fn eigen_normalize(inputs: [*]T, outputs: [*]T, aIn: [*]const T, bIn: [*]const T, aOut: [*]const T, bOut: [*]const T, nIn: usize, nOut: usize, nData: usize) void;
extern "c" fn eigen_denormalize(inputs: [*]T, outputs: [*]T, aIn: [*]const T, bIn: [*]const T, aOut: [*]const T, bOut: [*]const T, nIn: usize, nOut: usize, nData: usize) void;

/// Computes Z-score factors over a batch: writes the population std into `aIn`/`aOut`
/// and the mean into `bIn`/`bOut` using Eigen.
pub fn computeMeanStd(inputs: []const T, outputs: []const T, aIn: []T, bIn: []T, aOut: []T, bOut: []T) void {
    const nIn: usize = aIn.len;
    const nOut: usize = aOut.len;
    const nData: usize = inputs.len / nIn;
    eigen_computeMeanStd(inputs.ptr, outputs.ptr, aIn.ptr, bIn.ptr, aOut.ptr, bOut.ptr, nIn, nOut, nData);
}

test "[eigen] computeMeanStd" {
    const inputs = [_]T{ 0.0, 2.0, 4.0, 6.0, 8.0, 10.0 };
    const outputs = [_]T{ 1.0, 3.0, 5.0 };
    var aIn: [2]T = undefined;
    var bIn: [2]T = undefined;
    var aOut: [1]T = undefined;
    var bOut: [1]T = undefined;

    computeMeanStd(&inputs, &outputs, &aIn, &bIn, &aOut, &bOut);

    // feature 0: {0,4,8} mean=4 ; feature 1: {2,6,10} mean=6
    try testing.expectApproxEqAbs(4.0, bIn[0], 1e-4);
    try testing.expectApproxEqAbs(6.0, bIn[1], 1e-4);
    // std = sqrt(32/3) = 3.265986 for both input features
    try testing.expectApproxEqAbs(3.265986, aIn[0], 1e-4);
    try testing.expectApproxEqAbs(3.265986, aIn[1], 1e-4);
    // output feature {1,3,5}: mean=3, std=sqrt(8/3)=1.632993
    try testing.expectApproxEqAbs(3.0, bOut[0], 1e-4);
    try testing.expectApproxEqAbs(1.632993, aOut[0], 1e-4);
}

/// Normalizes `inputs`/`outputs` in place over a full batch: x' = (x - b) / a
pub fn normalize(inputs: []T, outputs: []T, aIn: []const T, bIn: []const T, aOut: []const T, bOut: []const T) void {
    const nIn: usize = aIn.len;
    const nOut: usize = aOut.len;
    const nData: usize = inputs.len / nIn;
    eigen_normalize(inputs.ptr, outputs.ptr, aIn.ptr, bIn.ptr, aOut.ptr, bOut.ptr, nIn, nOut, nData);
}

test "[eigen] normalize" {
    const aIn = [_]T{ 2.0, 4.0 };
    const bIn = [_]T{ 1.0, 2.0 };
    const aOut = [_]T{10.0};
    const bOut = [_]T{5.0};

    var inputs = [_]T{ 1.0, 2.0, 3.0, 6.0 };
    var outputs = [_]T{ 5.0, 15.0 };

    normalize(&inputs, &outputs, &aIn, &bIn, &aOut, &bOut);

    // inputs: ((1-1)/2,(2-2)/4, (3-1)/2,(6-2)/4) = (0,0,1,1)
    try testing.expectApproxEqAbs(0.0, inputs[0], 1e-4);
    try testing.expectApproxEqAbs(0.0, inputs[1], 1e-4);
    try testing.expectApproxEqAbs(1.0, inputs[2], 1e-4);
    try testing.expectApproxEqAbs(1.0, inputs[3], 1e-4);
    // outputs: ((5-5)/10, (15-5)/10) = (0,1)
    try testing.expectApproxEqAbs(0.0, outputs[0], 1e-4);
    try testing.expectApproxEqAbs(1.0, outputs[1], 1e-4);
}

/// Denormalizes `inputs`/`outputs` in place over a full batch: x = x' * a + b
pub fn denormalize(inputs: []T, outputs: []T, aIn: []const T, bIn: []const T, aOut: []const T, bOut: []const T) void {
    const nIn: usize = aIn.len;
    const nOut: usize = aOut.len;
    const nData: usize = inputs.len / nIn;
    eigen_denormalize(inputs.ptr, outputs.ptr, aIn.ptr, bIn.ptr, aOut.ptr, bOut.ptr, nIn, nOut, nData);
}

test "[eigen] denormalize" {
    const aIn = [_]T{ 2.0, 4.0 };
    const bIn = [_]T{ 1.0, 2.0 };
    const aOut = [_]T{10.0};
    const bOut = [_]T{5.0};

    // start from normalized values
    var inputs = [_]T{ 0.0, 0.0, 1.0, 1.0 };
    var outputs = [_]T{ 0.0, 1.0 };

    denormalize(&inputs, &outputs, &aIn, &bIn, &aOut, &bOut);

    // inputs back to {1,2,3,6}
    try testing.expectApproxEqAbs(1.0, inputs[0], 1e-4);
    try testing.expectApproxEqAbs(2.0, inputs[1], 1e-4);
    try testing.expectApproxEqAbs(3.0, inputs[2], 1e-4);
    try testing.expectApproxEqAbs(6.0, inputs[3], 1e-4);
    // outputs back to {5,15}
    try testing.expectApproxEqAbs(5.0, outputs[0], 1e-4);
    try testing.expectApproxEqAbs(15.0, outputs[1], 1e-4);
}
