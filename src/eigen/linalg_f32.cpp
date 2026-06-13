#include <Eigen/Dense>

extern "C"{

    // Multiply a matrix and a vector then add another vector
    void eigenf32_matrixVectorMulAdd(const float* a, const float* b, const float* c, float* result, size_t a_rows, size_t a_cols){
        Eigen::Map<const Eigen::MatrixXf> matA(a, a_rows, a_cols);
        Eigen::Map<const Eigen::VectorXf> vecB(b, a_cols);
        Eigen::Map<const Eigen::VectorXf> vecC(c, a_rows);
        Eigen::Map<Eigen::VectorXf> vecResult(result, a_rows);

        vecResult.noalias() = matA * vecB + vecC;
    }

    // Multiply a matrix and a batch of vectors then add a vector
    void eigenf32_matrixVectorMulAddBatch(const float* a, const float* b, const float* c, float* result, size_t a_rows, size_t a_cols, size_t batch_size){
        Eigen::Map<const Eigen::MatrixXf> matA(a, a_rows, a_cols);
        Eigen::Map<const Eigen::MatrixXf> vecB(b, a_cols, batch_size);
        Eigen::Map<const Eigen::VectorXf> vecC(c, a_rows);
        Eigen::Map<Eigen::MatrixXf> vecResult(result, a_rows, batch_size);

        vecResult.noalias() = (matA * vecB).colwise() + vecC;
    }

    // Multiply a vector by a matrix
    void eigenf32_vectorMatrixMul(float* a, const float* b, size_t b_rows, size_t b_cols){
        Eigen::Map<const Eigen::MatrixXf> matB(b, b_rows, b_cols);
        Eigen::Map<Eigen::VectorXf> vecA(a, b_rows);
        Eigen::Map<Eigen::VectorXf> vecResult(a, b_cols);

        vecResult = vecA.transpose() * matB;
    }

    // Multiply a batch of row vectors by a matrix
    void eigenf32_vectorMatrixMulBatch(const float* a, float* result, const float* b, size_t b_rows, size_t b_cols, size_t batch_size){
        Eigen::Map<const Eigen::MatrixXf> matA(a, b_rows, batch_size);
        Eigen::Map<const Eigen::MatrixXf> matB(b, b_rows, b_cols);
        Eigen::Map<Eigen::MatrixXf> matResult(result, b_cols, batch_size);

        matResult.noalias() = matB.transpose() * matA;
    }

    // Multiply two vectors elementwise
    void eigenf32_vectorMul(const float* a, float* b, size_t size){
        Eigen::Map<const Eigen::ArrayXf> vecA(a, size);
        Eigen::Map<Eigen::ArrayXf> vecB(b, size);

        vecB *= vecA;
    }

    // Set an array to zero
    void eigenf32_setZero(float* a, size_t size){
        Eigen::Map<Eigen::VectorXf> vecA(a, size);
        vecA.setZero();
    }

    // Set a matrix to the identity
    void eigenf32_vectorInit(const float* a, float* result, size_t size){
        Eigen::Map<const Eigen::VectorXf> vecA(a, size);
        Eigen::Map<Eigen::VectorXf> vecResult(result, size);

        vecResult = vecA;
    }

    // Update mt and vt for the weights
    void eigenf32_updateGradWeights(const float* v, const float* y, float* grad, size_t v_size, size_t y_size){
        Eigen::Map<const Eigen::VectorXf> vecV(v, v_size);
        Eigen::Map<const Eigen::VectorXf> vecY(y, y_size);
        Eigen::Map<Eigen::MatrixXf> matGrad(grad, v_size, y_size);

        // Update the gradients
        matGrad.noalias() += vecV * vecY.transpose();
    }

    // Update mt and vt for the weights (batch)
    void eigenf32_updateGradWeightsBatch(const float* v, const float* y, float* grad, size_t v_size, size_t y_size, size_t batch_size){
        Eigen::Map<const Eigen::MatrixXf> matV(v, v_size, batch_size);
        Eigen::Map<const Eigen::MatrixXf> matY(y, y_size, batch_size);
        Eigen::Map<Eigen::MatrixXf> matGrad(grad, v_size, y_size);

        matGrad.noalias() += matV * matY.transpose();
    }

    // Update mt and vt for the biases
    void eigenf32_updateGradBiases(const float* v, float* grad, size_t v_size){
        Eigen::Map<const Eigen::VectorXf> vecV(v, v_size);
        Eigen::Map<Eigen::VectorXf> vecGrad(grad, v_size);

        // Update the gradients
        vecGrad.noalias() += vecV;
    }

    // Update mt and vt for the biases (batch)
    void eigenf32_updateGradBiasesBatch(const float* v, float* grad, size_t v_size, size_t batch_size){
        Eigen::Map<const Eigen::MatrixXf> matV(v, v_size, batch_size);
        Eigen::Map<Eigen::VectorXf> vecGrad(grad, v_size);

        vecGrad += matV.rowwise().sum();
    }

}
