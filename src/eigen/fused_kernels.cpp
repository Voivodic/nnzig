#include <Eigen/Dense>
#include <regex>

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
using ArrayXt = Eigen::Array<f_type, Eigen::Dynamic, 1>;

extern "C" {

void eigen_forwardNone(const f_type *matrix, const f_type *vecs_mul,
                       const f_type *vec_sum, f_type *vecs_result,
                       size_t a_rows, size_t a_cols, size_t batch_size) {
  Eigen::Map<const MatrixXt> matA(matrix, a_rows, a_cols);
  Eigen::Map<const MatrixXt> matB(vecs_mul, a_cols, batch_size);
  Eigen::Map<const VectorXt> vecC(vec_sum, a_rows);
  Eigen::Map<MatrixXt> matResult(vecs_result, a_rows, batch_size);

  matResult.noalias() = (matA * matB).colwise() + vecC;
}

void eigen_forwardNoneGrad(const f_type *matrix, const f_type *vecs_mul,
                       const f_type *vec_sum, f_type *vecs_result,
                       f_type *vecs_deriv, size_t a_rows, size_t a_cols,
                       size_t batch_size) {
  Eigen::Map<const MatrixXt> matA(matrix, a_rows, a_cols);
  Eigen::Map<const MatrixXt> matB(vecs_mul, a_cols, batch_size);
  Eigen::Map<const VectorXt> vecC(vec_sum, a_rows);
  Eigen::Map<MatrixXt> matResult(vecs_result, a_rows, batch_size);
  Eigen::Map<ArrayXt> matDeriv(vecs_deriv, a_rows, batch_size);

  matResult.noalias() = (matA * matB).colwise() + vecC;
  matDeriv = f_type(1.0);
}
void eigen_forwardReLu(const f_type *matrix, const f_type *vecs_mul,
                       const f_type *vec_sum, f_type *vecs_result,
                       size_t a_rows, size_t a_cols, size_t batch_size) {
  Eigen::Map<const MatrixXt> matA(matrix, a_rows, a_cols);
  Eigen::Map<const MatrixXt> matB(vecs_mul, a_cols, batch_size);
  Eigen::Map<const VectorXt> vecC(vec_sum, a_rows);
  Eigen::Map<MatrixXt> matResult(vecs_result, a_rows, batch_size);

  matResult.noalias() = ((matA * matB).colwise() + vecC).cwiseMax(f_type(0.0));
}

void eigen_forwardReLuGrad(const f_type *matrix, const f_type *vecs_mul,
                       const f_type *vec_sum, f_type *vecs_result,
                       f_type *vecs_deriv, size_t a_rows, size_t a_cols,
                       size_t batch_size) {
  Eigen::Map<const MatrixXt> matA(matrix, a_rows, a_cols);
  Eigen::Map<const MatrixXt> matB(vecs_mul, a_cols, batch_size);
  Eigen::Map<const VectorXt> vecC(vec_sum, a_rows);
  Eigen::Map<MatrixXt> matResult(vecs_result, a_rows, batch_size);
  Eigen::Map<MatrixXt> matDeriv(vecs_deriv, a_rows, batch_size);

  matResult.noalias() = ((matA * matB).colwise() + vecC).cwiseMax(f_type(0.0));
  matDeriv = ((matResult.array() > 0.0).cast<f_type>()).matrix();
}
}
