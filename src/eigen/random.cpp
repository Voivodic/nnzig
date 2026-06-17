#include <Eigen/Dense>
#include <random>

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

extern "C"{

    // Fill data[0..size) with i.i.d. N(0, 1) samples using a deterministic
    // Mersenne-twister PRNG seeded by `seed`.  The PRNG lives in the backend
    // so that a future CUDA backend can use cuRAND without changing the call site.
    void eigen_initNormal(f_type* data, const size_t& size, const uint64_t& seed){
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);

        Eigen::Map<ArrayXt> arr(data, size);
        for(Eigen::Index i = 0; i < arr.size(); ++i){
            arr(i) = static_cast<f_type>(dist(rng));
        }
    }

}
