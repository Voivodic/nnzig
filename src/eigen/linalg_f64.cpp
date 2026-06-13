#include <Eigen/Dense>

extern "C"{

    // Multiply a matrix and a vector then add another vector
    void eigenf64_matrixVectorMulAdd(const double* a, const double* b, const double* c, double* result, size_t a_rows, size_t a_cols){
        Eigen::Map<const Eigen::MatrixXd> matA(a, a_rows, a_cols);
        Eigen::Map<const Eigen::VectorXd> vecB(b, a_cols);
        Eigen::Map<const Eigen::VectorXd> vecC(c, a_rows);
        Eigen::Map<Eigen::VectorXd> vecResult(result, a_rows);

        vecResult.noalias() = matA * vecB + vecC;
    }

    // Multiply a matrix and a batch of vectors then add a vector
    void eigenf64_matrixVectorMulAddBatch(const double* a, const double* b, const double* c, double* result, size_t a_rows, size_t a_cols, size_t batch_size){
        Eigen::Map<const Eigen::MatrixXd> matA(a, a_rows, a_cols);
        Eigen::Map<const Eigen::MatrixXd> matB(b, a_cols, batch_size);
        Eigen::Map<const Eigen::VectorXd> vecC(c, a_rows);
        Eigen::Map<Eigen::MatrixXd> matResult(result, a_rows, batch_size);

        matResult.noalias() = (matA * matB).colwise() + vecC;
    }

    // Multiply a vector by a matrix
    void eigenf64_vectorMatrixMul(double* a, const double* b, size_t b_rows, size_t b_cols){
        Eigen::Map<const Eigen::MatrixXd> matB(b, b_rows, b_cols);
        Eigen::Map<Eigen::VectorXd> vecA(a, b_rows);
        Eigen::Map<Eigen::VectorXd> vecResult(a, b_cols);

        vecResult = vecA.transpose() * matB;
    }

    // Multiply a batch of row vectors by a matrix
    void eigenf64_vectorMatrixMulBatch(const double* a, double* result, const double* b, size_t b_rows, size_t b_cols, size_t batch_size){
        Eigen::Map<const Eigen::MatrixXd> matA(a, b_rows, batch_size);
        Eigen::Map<const Eigen::MatrixXd> matB(b, b_rows, b_cols);
        Eigen::Map<Eigen::MatrixXd> matResult(result, b_cols, batch_size);

        matResult.noalias() = matB.transpose() * matA;
    }

    // Multiply two vectors elementwise
    void eigenf64_vectorMul(const double* a, double* b, size_t size){
        Eigen::Map<const Eigen::ArrayXd> vecA(a, size);
        Eigen::Map<Eigen::ArrayXd> vecB(b, size);

        vecB *= vecA;
    }

    // Set an array to zero
    void eigenf64_setZero(double* a, size_t size){
        Eigen::Map<Eigen::VectorXd> vecA(a, size);
        vecA.setZero();
    }

    // Set a matrix to the identity
    void eigenf64_vectorInit(const double* a, double* result, size_t size){
        Eigen::Map<const Eigen::VectorXd> vecA(a, size);
        Eigen::Map<Eigen::VectorXd> vecResult(result, size);

        vecResult = vecA;
    }

    // Update mt and vt for the weights
    void eigenf64_updateGradWeights(const double* v, const double* y, double* grad, size_t v_size, size_t y_size){
        Eigen::Map<const Eigen::VectorXd> vecV(v, v_size);
        Eigen::Map<const Eigen::VectorXd> vecY(y, y_size);
        Eigen::Map<Eigen::MatrixXd> matGrad(grad, v_size, y_size);

        // Update the gradients
        matGrad.noalias() += vecV * vecY.transpose();
    }

    // Update mt and vt for the weights (batch)
    void eigenf64_updateGradWeightsBatch(const double* v, const double* y, double* grad, size_t v_size, size_t y_size, size_t batch_size){
        Eigen::Map<const Eigen::MatrixXd> matV(v, v_size, batch_size);
        Eigen::Map<const Eigen::MatrixXd> matY(y, y_size, batch_size);
        Eigen::Map<Eigen::MatrixXd> matGrad(grad, v_size, y_size);

        matGrad.noalias() += matV * matY.transpose();
    }

    // Update mt and vt for the biases
    void eigenf64_updateGradBiases(const double* v, double* grad, size_t v_size){
        Eigen::Map<const Eigen::VectorXd> vecV(v, v_size);
        Eigen::Map<Eigen::VectorXd> vecGrad(grad, v_size);

        // Update the gradients
        vecGrad.noalias() += vecV;
    }

    // Update mt and vt for the biases (batch)
    void eigenf64_updateGradBiasesBatch(const double* v, double* grad, size_t v_size, size_t batch_size){
        Eigen::Map<const Eigen::MatrixXd> matV(v, v_size, batch_size);
        Eigen::Map<Eigen::VectorXd> vecGrad(grad, v_size);

        vecGrad += matV.rowwise().sum();
    }

}
