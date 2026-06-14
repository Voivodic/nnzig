#include <Eigen/Dense>
#include <cmath>

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

    // None activation: set all derivatives to 1.0, leave the input untouched
    void eigen_none(f_type* input, f_type* df, size_t n){
        (void)input;
        Eigen::Map<ArrayXt> dfMap(df, n);
        dfMap = f_type(1.0);
    }

    // ReLU activation and its derivative, computed element-wise over the whole array
    void eigen_relu(f_type* input, f_type* df, size_t n){
        Eigen::Map<ArrayXt> inMap(input, n);
        Eigen::Map<ArrayXt> dfMap(df, n);

        // Keep a copy of the original values so the derivative is correct
        ArrayXt orig = inMap;
        dfMap = (orig >= f_type(0.0)).cast<f_type>();
        inMap = orig.cwiseMax(f_type(0.0));
    }

    // Hyperbolic tangent activation and its derivative, computed element-wise
    void eigen_tanh(f_type* input, f_type* df, size_t n){
        Eigen::Map<ArrayXt> inMap(input, n);
        Eigen::Map<ArrayXt> dfMap(df, n);

        ArrayXt t = inMap.tanh();
        dfMap = f_type(1.0) - t * t;
        inMap = t;
    }

    // Sigmoid activation and its derivative, computed element-wise
    void eigen_sigmoid(f_type* input, f_type* df, size_t n){
        Eigen::Map<ArrayXt> inMap(input, n);
        Eigen::Map<ArrayXt> dfMap(df, n);

        ArrayXt s = f_type(1.0) / (f_type(1.0) + (-inMap).exp());
        dfMap = s * (f_type(1.0) - s);
        inMap = s;
    }

}
