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

// Define the functions that will be exported to zig
extern "C"{

    // Compute Z-score normalization factors: a = population std, b = mean.
    // Data is row-major (sample i then feature j); mapped as (nDim x nData) so each column is a sample.
    void eigen_computeMeanStd(const f_type* inputs, const f_type* outputs,
                              f_type* aIn, f_type* bIn, f_type* aOut, f_type* bOut,
                              size_t nIn, size_t nOut, size_t nData){
        {
            Eigen::Map<const MatrixXt> matIn(inputs, nIn, nData);
            Eigen::Map<VectorXt> bInVec(bIn, nIn);
            Eigen::Map<VectorXt> aInVec(aIn, nIn);
            bInVec = matIn.rowwise().mean();
            MatrixXt centered = matIn.colwise() - bInVec;
            aInVec = (centered.array().square().rowwise().mean()).sqrt();
        }
        {
            Eigen::Map<const MatrixXt> matOut(outputs, nOut, nData);
            Eigen::Map<VectorXt> bOutVec(bOut, nOut);
            Eigen::Map<VectorXt> aOutVec(aOut, nOut);
            bOutVec = matOut.rowwise().mean();
            MatrixXt centered = matOut.colwise() - bOutVec;
            aOutVec = (centered.array().square().rowwise().mean()).sqrt();
        }
    }

    // Normalize in place over a batch of data points: x' = (x - b) / a
    void eigen_normalize(f_type* inputs, f_type* outputs,
                         const f_type* aIn, const f_type* bIn, const f_type* aOut, const f_type* bOut,
                         size_t nIn, size_t nOut, size_t nData){
        {
            Eigen::Map<MatrixXt> matIn(inputs, nIn, nData);
            Eigen::Map<const VectorXt> bInVec(bIn, nIn);
            Eigen::Map<const VectorXt> aInVec(aIn, nIn);
            matIn.colwise() -= bInVec;
            matIn = aInVec.cwiseInverse().asDiagonal() * matIn;
        }
        {
            Eigen::Map<MatrixXt> matOut(outputs, nOut, nData);
            Eigen::Map<const VectorXt> bOutVec(bOut, nOut);
            Eigen::Map<const VectorXt> aOutVec(aOut, nOut);
            matOut.colwise() -= bOutVec;
            matOut = aOutVec.cwiseInverse().asDiagonal() * matOut;
        }
    }

    // Denormalize in place over a batch of data points: x = x' * a + b
    void eigen_denormalize(f_type* inputs, f_type* outputs,
                           const f_type* aIn, const f_type* bIn, const f_type* aOut, const f_type* bOut,
                           size_t nIn, size_t nOut, size_t nData){
        {
            Eigen::Map<MatrixXt> matIn(inputs, nIn, nData);
            Eigen::Map<const VectorXt> bInVec(bIn, nIn);
            Eigen::Map<const VectorXt> aInVec(aIn, nIn);
            matIn = aInVec.asDiagonal() * matIn;
            matIn.colwise() += bInVec;
        }
        {
            Eigen::Map<MatrixXt> matOut(outputs, nOut, nData);
            Eigen::Map<const VectorXt> bOutVec(bOut, nOut);
            Eigen::Map<const VectorXt> aOutVec(aOut, nOut);
            matOut = aOutVec.asDiagonal() * matOut;
            matOut.colwise() += bOutVec;
        }
    }

}
