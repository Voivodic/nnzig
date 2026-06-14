#include <Eigen/Dense>

#ifndef FLOAT_PRECISION
#define FLOAT_PRECISION 32
#endif

#if FLOAT_PRECISION == 16
    using f_type = Eigen::half;
#elif FLOAT_PRECISION == 32
    using f_type = float;
#elif FLOAT_PRECISION == 64
    using f_type = double;
#else
    #error "FLOAT_PRECISION must be 16, 32, or 64"
#endif

using MatrixXt = Eigen::Matrix<f_type, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXt = Eigen::Matrix<f_type, Eigen::Dynamic, 1>;
using ArrayXt  = Eigen::Array<f_type, Eigen::Dynamic, 1>;

extern "C"{

    // Multiply a matrix and a vector then add another vector
    void eigen_matrixVectorMulAdd(const f_type* a, const f_type* b, const f_type* c, f_type* result, size_t a_rows, size_t a_cols){
        Eigen::Map<const MatrixXt> matA(a, a_rows, a_cols);
        Eigen::Map<const VectorXt> vecB(b, a_cols);
        Eigen::Map<const VectorXt> vecC(c, a_rows);
        Eigen::Map<VectorXt> vecResult(result, a_rows);

        vecResult.noalias() = matA * vecB + vecC;
    }

    // Multiply a matrix and a batch of vectors then add a vector
    void eigen_matrixVectorMulAddBatch(const f_type* a, const f_type* b, const f_type* c, f_type* result, size_t a_rows, size_t a_cols, size_t batch_size){
        Eigen::Map<const MatrixXt> matA(a, a_rows, a_cols);
        Eigen::Map<const MatrixXt> matB(b, a_cols, batch_size);
        Eigen::Map<const VectorXt> vecC(c, a_rows);
        Eigen::Map<MatrixXt> matResult(result, a_rows, batch_size);

        matResult.noalias() = (matA * matB).colwise() + vecC;
    }

    // Multiply a vector by a matrix
    void eigen_vectorMatrixMul(f_type* a, const f_type* b, size_t b_rows, size_t b_cols){
        Eigen::Map<const MatrixXt> matB(b, b_rows, b_cols);
        Eigen::Map<VectorXt> vecA(a, b_rows);
        Eigen::Map<VectorXt> vecResult(a, b_cols);

        vecResult = vecA.transpose() * matB;
    }

    // Multiply a batch of row vectors by a matrix
    void eigen_vectorMatrixMulBatch(const f_type* a, f_type* result, const f_type* b, size_t b_rows, size_t b_cols, size_t batch_size){
        Eigen::Map<const MatrixXt> matA(a, b_rows, batch_size);
        Eigen::Map<const MatrixXt> matB(b, b_rows, b_cols);
        Eigen::Map<MatrixXt> matResult(result, b_cols, batch_size);

        matResult.noalias() = matB.transpose() * matA;
    }

    // Multiply two vectors elementwise
    void eigen_vectorMul(const f_type* a, f_type* b, size_t size){
        Eigen::Map<const ArrayXt> vecA(a, size);
        Eigen::Map<ArrayXt> vecB(b, size);

        vecB *= vecA;
    }

    // Set an array to zero
    void eigen_setZero(f_type* a, size_t size){
        Eigen::Map<VectorXt> vecA(a, size);
        vecA.setZero();
    }

    // Copy all elements from one vector into another
    void eigen_vectorInit(const f_type* a, f_type* result, size_t size){
        Eigen::Map<const VectorXt> vecA(a, size);
        Eigen::Map<VectorXt> vecResult(result, size);

        vecResult = vecA;
    }

    // Accumulate the outer product of v and y into the weight gradient
    void eigen_updateGradWeights(const f_type* v, const f_type* y, f_type* grad, size_t v_size, size_t y_size){
        Eigen::Map<const VectorXt> vecV(v, v_size);
        Eigen::Map<const VectorXt> vecY(y, y_size);
        Eigen::Map<MatrixXt> matGrad(grad, v_size, y_size);

        matGrad.noalias() += vecV * vecY.transpose();
    }

    // Accumulate outer products from a batch into the weight gradient
    void eigen_updateGradWeightsBatch(const f_type* v, const f_type* y, f_type* grad, size_t v_size, size_t y_size, size_t batch_size){
        Eigen::Map<const MatrixXt> matV(v, v_size, batch_size);
        Eigen::Map<const MatrixXt> matY(y, y_size, batch_size);
        Eigen::Map<MatrixXt> matGrad(grad, v_size, y_size);

        matGrad.noalias() += matV * matY.transpose();
    }

    // Add v element-wise into the bias gradient
    void eigen_updateGradBiases(const f_type* v, f_type* grad, size_t v_size){
        Eigen::Map<const VectorXt> vecV(v, v_size);
        Eigen::Map<VectorXt> vecGrad(grad, v_size);

        vecGrad.noalias() += vecV;
    }

    // Accumulate bias gradients from a batch
    void eigen_updateGradBiasesBatch(const f_type* v, f_type* grad, size_t v_size, size_t batch_size){
        Eigen::Map<const MatrixXt> matV(v, v_size, batch_size);
        Eigen::Map<VectorXt> vecGrad(grad, v_size);

        vecGrad += matV.rowwise().sum();
    }

}
