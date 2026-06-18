//! Zig wrappers around Eigen C++ linear algebra routines. Provides functions
//! for matrix-vector operations used in neural network forward and backward
//! passes. The float precision is selected at compile time via params.T.
const testing = @import("std").testing;
const params = @import("params");

const T = params.T;

extern "c" fn eigen_initThreads() void;

/// Initializes the number of threads used by Eigen
pub fn initThreads() void {
    eigen_initThreads();
}

// --- Matrix * vector + vector ---


extern "c" fn eigen_matrixVectorMulAdd(matrix: [*]const T, vecs_mul: [*]const T, vec_sum: [*]const T, vecs_result: [*]T, a_rows: usize, a_cols: usize, batch_size: usize) void;

/// Computes `result = matrix * vector_mult + vector_add` using Eigen for a full batch
pub fn matrixVectorMulAdd(matrix: []const T, vectorsMult: []const T, vectorAdd: []const T, vectorResult: []T) void {
    const dimOut: usize = vectorAdd.len;
    const dimIn: usize = matrix.len / dimOut;
    const batchSize: usize = vectorsMult.len / dimIn;

    eigen_matrixVectorMulAdd(matrix.ptr, vectorsMult.ptr, vectorAdd.ptr, vectorResult.ptr, dimOut, dimIn, batchSize);
}

test "[eigen] matrixVectorMulAdd" {
    const matrix = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    const vMult = [_]T{ 1.0, 2.0, 1.0, 2.0 };
    const vAdd = [_]T{ 5.0, 6.0 };
    var result = [_]T{ 0.0, 0.0, 0.0, 0.0 };

    matrixVectorMulAdd(matrix[0..], vMult[0..], vAdd[0..], result[0..]);

    try testing.expectApproxEqAbs(12.0, result[0], 0.001);
    try testing.expectApproxEqAbs(16.0, result[1], 0.001);
    try testing.expectApproxEqAbs(12.0, result[2], 0.001);
    try testing.expectApproxEqAbs(16.0, result[3], 0.001);
}

// --- Vector * matrix ---

extern "c" fn eigen_vectorMatrixMul(vecs: [*]T, mat: [*]const T, b_rows: usize, b_cols: usize, batch_size: usize) void;

/// Computes `result = vectors^T * matrix` for a batch of row vectors using Eigen
pub fn vectorMatrixMul(vectors: []T, matrix: []const T, batchSize: usize) void {
    const dimIn: usize = vectors.len / batchSize;
    const dimOut: usize = matrix.len / dimIn;

    eigen_vectorMatrixMul(vectors.ptr, matrix.ptr, dimIn, dimOut, batchSize);
}

test "[eigen] vectorMatrixMul" {
    var vectors = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    const matrix = [_]T{ 1.0, 2.0, 3.0, 4.0 };

    vectorMatrixMul(vectors[0..], matrix[0..], 2);

    try testing.expectApproxEqAbs(5.0, vectors[0], 0.001);
    try testing.expectApproxEqAbs(11.0, vectors[1], 0.001);
    try testing.expectApproxEqAbs(11.0, vectors[2], 0.001);
    try testing.expectApproxEqAbs(25.0, vectors[3], 0.001);
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

extern "c" fn eigen_updateGradWeights(V: [*]const T, Y: [*]const T, grad: [*]T, v_size: usize, y_size: usize, batch_size: usize) void;

/// Accumulates the outer products from a batch into the weight gradient (`grad += V * Y^T`)
pub fn updateGradWeights(v: []const T, y: []const T, grad: []T, batchSize: usize) void {
    const dimV: usize = v.len / batchSize;
    const dimY: usize = y.len / batchSize;

    eigen_updateGradWeights(v.ptr, y.ptr, grad.ptr, dimV, dimY, batchSize);
}

test "[eigen] updateGradWeights" {
    const v = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    const y = [_]T{ 0.5, 0.5, 1.0, 1.0 };
    var grad = [_]T{ 0.0, 0.0, 0.0, 0.0 };

    updateGradWeights(v[0..], y[0..], grad[0..], 2);

    try testing.expectApproxEqAbs(3.5, grad[0], 0.001);
    try testing.expectApproxEqAbs(5.0, grad[1], 0.001);
    try testing.expectApproxEqAbs(3.5, grad[2], 0.001);
    try testing.expectApproxEqAbs(5.0, grad[3], 0.001);
}

// --- Gradient accumulation (biases) ---

extern "c" fn eigen_updateGradBiases(V: [*]const T, grad: [*]T, v_size: usize, batch_size: usize) void;

/// Accumulates bias gradients from a batch (`grad += rowwise_sum(V)`)
pub fn updateGradBiases(v: []const T, grad: []T) void {
    const v_size: usize = grad.len;
    const batch_size: usize = v.len / v_size;

    eigen_updateGradBiases(v.ptr, grad.ptr, v_size, batch_size);
}

test "[eigen] updateGradBiases" {
    const v = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    var grad = [_]T{ 1.0, 1.0 };

    updateGradBiases(v[0..], grad[0..]);

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

    try testing.expectApproxEqAbs(-0.964028, input[0], 1e-3);
    try testing.expectApproxEqAbs(-0.761594, input[1], 1e-3);
    try testing.expectApproxEqAbs(0.0, input[2], 1e-3);
    try testing.expectApproxEqAbs(0.761594, input[3], 1e-3);
    try testing.expectApproxEqAbs(0.964028, input[4], 1e-3);

    try testing.expectApproxEqAbs(0.0706508, df[0], 1e-3);
    try testing.expectApproxEqAbs(0.419974, df[1], 1e-3);
    try testing.expectApproxEqAbs(1.0, df[2], 1e-3);
    try testing.expectApproxEqAbs(0.419974, df[3], 1e-3);
    try testing.expectApproxEqAbs(0.0706508, df[4], 1e-3);
}

/// Applies sigmoid element-wise to `input` (in place) and writes its derivative into `df`
pub fn activateSigmoid(input: []T, df: []T) void {
    eigen_sigmoid(input.ptr, df.ptr, input.len);
}

test "[eigen] activateSigmoid" {
    var input = [_]T{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var df: [5]T = undefined;

    activateSigmoid(&input, &df);

    try testing.expectApproxEqAbs(0.119203, input[0], 1e-3);
    try testing.expectApproxEqAbs(0.268941, input[1], 1e-3);
    try testing.expectApproxEqAbs(0.5, input[2], 1e-3);
    try testing.expectApproxEqAbs(0.731059, input[3], 1e-3);
    try testing.expectApproxEqAbs(0.880797, input[4], 1e-3);

    try testing.expectApproxEqAbs(0.104994, df[0], 1e-3);
    try testing.expectApproxEqAbs(0.196612, df[1], 1e-3);
    try testing.expectApproxEqAbs(0.25, df[2], 1e-3);
    try testing.expectApproxEqAbs(0.196612, df[3], 1e-3);
    try testing.expectApproxEqAbs(0.104994, df[4], 1e-3);
}

// --- Loss functions ---

extern "c" fn eigen_mse(pred: [*]const T, out: [*]const T, dL: [*]T, loss: *T, n: usize) void;

/// Computes the mean squared error `L = 0.5 * sum((pred - out)^2)` and writes the
/// derivative `dL = (pred - out)` using Eigen. Returns the scalar loss.
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

    try testing.expectApproxEqAbs(8.0, lossVal, 1e-3);
    for (dL) |val| try testing.expectApproxEqAbs(2.0, val, 1e-3);

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
    try testing.expectApproxEqAbs(4.0, bIn[0], 1e-3);
    try testing.expectApproxEqAbs(6.0, bIn[1], 1e-3);
    // std = sqrt(32/3) = 3.265986 for both input features
    try testing.expectApproxEqAbs(3.265986, aIn[0], 1e-3);
    try testing.expectApproxEqAbs(3.265986, aIn[1], 1e-3);
    // output feature {1,3,5}: mean=3, std=sqrt(8/3)=1.632993
    try testing.expectApproxEqAbs(3.0, bOut[0], 1e-3);
    try testing.expectApproxEqAbs(1.632993, aOut[0], 1e-3);
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
    try testing.expectApproxEqAbs(0.0, inputs[0], 1e-3);
    try testing.expectApproxEqAbs(0.0, inputs[1], 1e-3);
    try testing.expectApproxEqAbs(1.0, inputs[2], 1e-3);
    try testing.expectApproxEqAbs(1.0, inputs[3], 1e-3);
    // outputs: ((5-5)/10, (15-5)/10) = (0,1)
    try testing.expectApproxEqAbs(0.0, outputs[0], 1e-3);
    try testing.expectApproxEqAbs(1.0, outputs[1], 1e-3);
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
    try testing.expectApproxEqAbs(1.0, inputs[0], 1e-3);
    try testing.expectApproxEqAbs(2.0, inputs[1], 1e-3);
    try testing.expectApproxEqAbs(3.0, inputs[2], 1e-3);
    try testing.expectApproxEqAbs(6.0, inputs[3], 1e-3);
    // outputs back to {5,15}
    try testing.expectApproxEqAbs(5.0, outputs[0], 1e-3);
    try testing.expectApproxEqAbs(15.0, outputs[1], 1e-3);
}

// --- Random initialization ---

extern "c" fn eigen_initWeights(data: [*]T, dim: *const usize, sigma: *const T, seed: *const u64) void;

/// Fills `vec` with i.i.d. N(0, sigma) random values using a deterministic PRNG seeded by `seed`.
pub fn initWeights(vec: []T, sigma: T, seed: u64) void {
    eigen_initWeights(vec.ptr, &vec.len, &sigma, &seed);
}

// --- Adam optimizer step ---

extern "c" fn eigen_adamUpdate(w: [*]T, grad: [*]const T, m: [*]T, v: [*]T, size: *const usize, beta1: *const T, beta2: *const T, lr: *const T, eps: *const T, normM: *const T, normV: *const T) void;

/// Performs an element-wise Adam optimizer update in place:
/// `m = beta1*m + (1-beta1)*grad`, `v = beta2*v + (1-beta2)*grad^2`,
/// `w -= (m/normM) * lr / (sqrt(v/normV) + eps)`.
/// All scalar parameters are passed by reference to the backend.
pub fn adamUpdate(w: []T, grad: []const T, m: []T, v: []T, beta1: T, beta2: T, lr: T, eps: T, normM: T, normV: T) void {
    const size = w.len;
    eigen_adamUpdate(w.ptr, grad.ptr, m.ptr, v.ptr, &size, &beta1, &beta2, &lr, &eps, &normM, &normV);
}

test "[eigen] adamUpdate" {
    // Set up: w=1, grad=0.5, m=0, v=0
    // beta1=0.9, beta2=0.999, lr=0.1, eps=1e-8, normM=0.9, normV=0.999
    var w = [_]T{ 1.0, 1.0, 1.0 };
    const grad = [_]T{ 0.5, 0.5, 0.5 };
    var m = [_]T{ 0.0, 0.0, 0.0 };
    var v = [_]T{ 0.0, 0.0, 0.0 };

    adamUpdate(w[0..], grad[0..], m[0..], v[0..], 0.9, 0.999, 0.1, 1e-8, 0.9, 0.999);

    // m = 0.9*0 + 0.1*0.5 = 0.05
    // v = 0.999*0 + 0.001*0.25 = 0.00025
    // m_hat = 0.05 / 0.9 = 0.0556
    // v_hat = 0.00025 / 0.999 = 0.00025025
    // sqrt(v_hat) = 0.015819
    // update = m_hat * 0.1 / (0.015819 + 1e-8) = 0.005556 / 0.015819 = 0.35122
    // w_new = 1.0 - 0.35122 = 0.64878
    try testing.expectApproxEqAbs(@as(T, 0.05), m[0], 1e-5);
    try testing.expectApproxEqAbs(@as(T, 0.00025), v[0], 1e-7);
    try testing.expectApproxEqAbs(@as(T, 0.64878), w[0], 1e-3);
}

// --- Scalar division (gradient normalization) ---

extern "c" fn eigen_divScalar(a: [*]T, size: *const usize, divisor: *const T) void;

/// Divides every element of `vec` by `divisor` in place using Eigen.
pub fn divScalar(vec: []T, divisor: T) void {
    const size = vec.len;
    eigen_divScalar(vec.ptr, &size, &divisor);
}

test "[eigen] divScalar" {
    var v = [_]T{ 10.0, 20.0, 30.0, 40.0 };

    divScalar(v[0..], 4.0);

    try testing.expectApproxEqAbs(@as(T, 2.5), v[0], 1e-5);
    try testing.expectApproxEqAbs(@as(T, 5.0), v[1], 1e-5);
    try testing.expectApproxEqAbs(@as(T, 7.5), v[2], 1e-5);
    try testing.expectApproxEqAbs(@as(T, 10.0), v[3], 1e-5);
}
