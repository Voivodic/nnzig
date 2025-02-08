#include <Eigen/Dense>

extern "C"{

// Multiply a matrix and a vector then add another vector
void matrixVectorMulAdd(const float* a, const float* b, const float* c, float* result, size_t a_rows, size_t a_cols){
    Eigen::Map<const Eigen::MatrixXf> matA(a, a_rows, a_cols);
    Eigen::Map<const Eigen::VectorXf> vecB(b, a_cols);
    Eigen::Map<const Eigen::VectorXf> vecC(c, a_rows);
    Eigen::Map<Eigen::VectorXf> vecResult(result, a_rows);

    vecResult = matA * vecB + vecC;
}

// Multiply a vector by a matrix
void vectorMatrixMul(float* a, const float* b, size_t b_rows, size_t b_cols){
    Eigen::Map<const Eigen::MatrixXf> matA(b, b_rows, b_cols);
    Eigen::Map<Eigen::VectorXf> vecB(a, b_rows);
    Eigen::VectorXf vecResult(b_cols);

    vecResult = vecB.transpose() * matA;
    a = vecResult.data();
}

// Multiply two vectors elementwise
void vectorMul(const float* a, float* b, size_t size){
    Eigen::Map<const Eigen::VectorXf> vecA(a, size);
    Eigen::Map<Eigen::VectorXf> vecB(b, size);

    vecB = (vecA.array() * vecB.array()).matrix();
}

// Set an array to zero
void setZero(float* a, size_t size){
    Eigen::Map<Eigen::VectorXf> vecA(a, size);
    
    vecA.setZero();
}

// Set a matrix to the identity
void vectorVInit(const float* a, float* result, size_t size){
    Eigen::Map<const Eigen::VectorXf> vecA(a, size);
    Eigen::Map<Eigen::VectorXf> vecResult(result, size);

    vecResult = vecA;
}

// Update mt and vt for the weights
void updateGradWeights(const float* v, const float* y, float* grad, size_t v_size, size_t y_size){
    Eigen::Map<const Eigen::VectorXf> vecV(v, v_size);
    Eigen::Map<const Eigen::VectorXf> vecY(y, y_size);
    Eigen::Map<Eigen::MatrixXf> matGrad(grad, v_size, y_size);

    // Update the gradients
    matGrad = matGrad + vecV * vecY.transpose();
}

// Update mt and vt for the biases
void updateGradBiases(const float* v, float* grad, size_t v_size){
    Eigen::Map<const Eigen::VectorXf> vecV(v, v_size);
    Eigen::Map<Eigen::VectorXf> vecGrad(grad, v_size);

    // Update the gradients
    vecGrad = vecGrad + vecV;
}

}
