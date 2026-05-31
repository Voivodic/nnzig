#include <Eigen/Dense>

// Define custom types for half-precision floating point
using half     = Eigen::half;
using MatrixXh = Eigen::Matrix<half, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXh = Eigen::Matrix<half, Eigen::Dynamic, 1>;
using ArrayXh  = Eigen::Array<half, Eigen::Dynamic, 1>;

extern "C"{

    // Multiply a matrix and a vector then add another vector
    void eigenf16_matrixVectorMulAdd(const half* a, const half* b, const half* c, half* result, size_t a_rows, size_t a_cols){
        Eigen::Map<const MatrixXh> matA(a, a_rows, a_cols);
        Eigen::Map<const VectorXh> vecB(b, a_cols);
        Eigen::Map<const VectorXh> vecC(c, a_rows);
        Eigen::Map<VectorXh> vecResult(result, a_rows);

        vecResult.noalias() = matA * vecB + vecC;
    }

    // Multiply a vector by a matrix
    void eigenf16_vectorMatrixMul(half* a, const half* b, size_t b_rows, size_t b_cols){
        Eigen::Map<const MatrixXh> matB(b, b_rows, b_cols);
        Eigen::Map<VectorXh> vecA(a, b_rows);
        Eigen::Map<VectorXh> vecResult(a, b_cols);

        vecResult = vecA.transpose() * matB;
    }

    // Multiply two vectors elementwise
    void eigenf16_vectorMul(const half* a, half* b, size_t size){
        Eigen::Map<const ArrayXh> vecA(a, size);
        Eigen::Map<ArrayXh> vecB(b, size);

        vecB *= vecA;
    }

    // Set an array to zero
    void eigenf16_setZero(half* a, size_t size){
        Eigen::Map<VectorXh> vecA(a, size);
        vecA.setZero();
    }

    // Set a matrix to the identity
    void eigenf16_vectorInit(const half* a, half* result, size_t size){
        Eigen::Map<const VectorXh> vecA(a, size);
        Eigen::Map<VectorXh> vecResult(result, size);

        vecResult = vecA;
    }

    // Update mt and vt for the weights
    void eigenf16_updateGradWeights(const half* v, const half* y, half* grad, size_t v_size, size_t y_size){
        Eigen::Map<const VectorXh> vecV(v, v_size);
        Eigen::Map<const VectorXh> vecY(y, y_size);
        Eigen::Map<MatrixXh> matGrad(grad, v_size, y_size);

        // Update the gradients
        matGrad.noalias() += vecV * vecY.transpose();
    }

    // Update mt and vt for the biases
    void eigenf16_updateGradBiases(const half* v, half* grad, size_t v_size){
        Eigen::Map<const VectorXh> vecV(v, v_size);
        Eigen::Map<VectorXh> vecGrad(grad, v_size);

        // Update the gradients
        vecGrad.noalias() += vecV;
    }

}
