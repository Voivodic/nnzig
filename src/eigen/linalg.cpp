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

#ifndef NUM_THREADS
#define NUM_THREADS 1
#endif

using MatrixXt = Eigen::Matrix<f_type, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXt = Eigen::Matrix<f_type, Eigen::Dynamic, 1>;
using ArrayXt  = Eigen::Array<f_type, Eigen::Dynamic, 1>;

extern "C"{

    // Initialize the threads
    void eigen_initThreads(){
        Eigen::setNbThreads(NUM_THREADS);
    }

    // Multiply a matrix and a batch of vectors then add a vector
    void eigen_matrixVectorMulAdd(const f_type* matrix, const f_type* vecs_mul, const f_type* vec_sum, f_type* vecs_result, size_t a_rows, size_t a_cols, size_t batch_size){
        Eigen::Map<const MatrixXt> matA(matrix, a_rows, a_cols);
        Eigen::Map<const MatrixXt> matB(vecs_mul, a_cols, batch_size);
        Eigen::Map<const VectorXt> vecC(vec_sum, a_rows);
        Eigen::Map<MatrixXt> matResult(vecs_result, a_rows, batch_size);

        matResult.noalias() = (matA * matB).colwise() + vecC;
    }

    // Multiply a batch of row vectors by a matrix
    void eigen_vectorMatrixMul(f_type* vecs, const f_type* mat, size_t b_rows, size_t b_cols, size_t batch_size){
        Eigen::Map<const MatrixXt> matA(mat, b_rows, b_cols);
        Eigen::Map<MatrixXt> vecB(vecs, b_rows, batch_size);
        Eigen::Map<MatrixXt> vecResult(vecs, b_cols, batch_size);

        vecResult = (vecB.transpose() * matA).transpose();
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

    // Accumulate outer products from a batch into the weight gradient
    void eigen_updateGradWeights(const f_type* v, const f_type* y, f_type* grad, size_t v_size, size_t y_size, size_t batch_size){
        Eigen::Map<const MatrixXt> matV(v, v_size, batch_size);
        Eigen::Map<const MatrixXt> matY(y, y_size, batch_size);
        Eigen::Map<MatrixXt> matGrad(grad, v_size, y_size);

        matGrad.noalias() += matV * matY.transpose();
    }

    // Accumulate bias gradients from a batch
    void eigen_updateGradBiases(const f_type* v, f_type* grad, size_t v_size, size_t batch_size){
        Eigen::Map<const MatrixXt> matV(v, v_size, batch_size);
        Eigen::Map<VectorXt> vecGrad(grad, v_size);

        vecGrad += matV.rowwise().sum();
    }

    // Element-wise Adam optimizer step: updates moments m and v, then applies the
    // bias-corrected weight update in place.  All scalars are passed by reference
    // so a future CUDA backend can place them in constant/device memory.
    void eigen_adamUpdate(f_type* w, const f_type* grad, f_type* m, f_type* v,
                          const size_t& size,
                          const f_type& beta1, const f_type& beta2,
                          const f_type& lr, const f_type& eps,
                          const f_type& normM, const f_type& normV){
        Eigen::Map<ArrayXt> arrW(w, size);
        Eigen::Map<const ArrayXt> arrGrad(grad, size);
        Eigen::Map<ArrayXt> arrM(m, size);
        Eigen::Map<ArrayXt> arrV(v, size);

        // Update biased moments
        arrM = beta1 * arrM + (f_type(1.0) - beta1) * arrGrad;
        arrV = beta2 * arrV + (f_type(1.0) - beta2) * arrGrad.square();

        // Apply bias-corrected update
        arrW -= (arrM / normM) * lr / ((arrV / normV).sqrt() + eps);
    }

    // Divide every element by a scalar (used for gradient normalization)
    void eigen_divScalar(f_type* a, const size_t& size, const f_type& divisor){
        Eigen::Map<ArrayXt> arr(a, size);
        arr /= divisor;
    }

}
