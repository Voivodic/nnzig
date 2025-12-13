// Multiply a matrix and a vector then add another vector
pub extern "c" fn matrixVectorMulAdd(A: [*]const f32, B: [*]const f32, C: [*]const f32, D: [*]f32, a_rows: usize, a_cols: usize) void;

// Multiply a vector by a matrix
pub extern "c" fn vectorMatrixMul(A: [*]f32, B: [*]const f32, b_rows: usize, b_cols: usize) void;

// Multiply two vectors elementwise
pub extern "c" fn vectorMul(A: [*]const f32, B: [*]f32, size: usize) void;

// Set an array to zero
pub extern "c" fn setZero(A: [*]f32, size: usize) void;

// Set a matrix to the identity
pub extern "c" fn vectorVInit(A: [*]const f32, B: [*]f32, size: usize) void;

// Update mt and vt for the weights
pub extern "c" fn updateGradWeights(V: [*]const f32, Y: [*]const f32, grad: [*]f32, v_size: usize, y_size: usize) void;

// Update mt and vt for the biases
pub extern "c" fn updateGradBiases(V: [*]const f32, grad: [*]f32, v_size: usize) void;
