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

using ArrayXt = Eigen::Array<f_type, Eigen::Dynamic, 1>;

// Define the functions that will be exported to zig
extern "C"{

    // Mean squared error: L = 0.5 * mean((pred - out)^2), dL = (pred - out) / n
    void eigen_mse(const f_type* pred, const f_type* out, f_type* dL, f_type* loss, size_t n){
        Eigen::Map<const ArrayXt> aPred(pred, n);
        Eigen::Map<const ArrayXt> aOut(out, n);
        Eigen::Map<ArrayXt> aDl(dL, n);

        ArrayXt diff = aPred - aOut;
        aDl = diff / f_type(n);
        *loss = f_type(0.5) * diff.square().sum() / f_type(n);
    }

}
