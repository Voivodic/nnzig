#include <Eigen/Dense>
#include <cmath>
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

    // Kaiming/He variance-scaled initialization (forward/fan_in mode).
    // Fills data[0..fan_in*fan_out) with i.i.d. N(0, (gain / sqrt(fan_in))^2)
    // samples, where `gain` encodes the layer's activation function (sqrt(2)
    // for ReLU). This is the initializer of choice for ReLU layers: it keeps
    // the variance of forward activations roughly constant across layers,
    // unlike plain N(0,1) whose pre-activation variance Var(z) grows as
    // fan_in * Var(w). `fan_out` is accepted for signature symmetry with
    // eigen_initXavier but is not used (He forward mode depends on fan_in).
    //
    // As with eigen_initNormal, the PRNG lives in the backend so a future
    // CUDA backend can swap in cuRAND without changing the call site.
    void eigen_initKaiming(f_type* data, const size_t& fan_in, const size_t& fan_out,
                           const f_type& gain, const uint64_t& seed){
        std::mt19937_64 rng(seed);
        const double sigma = static_cast<double>(gain) / std::sqrt(static_cast<double>(fan_in));
        std::normal_distribution<double> dist(0.0, sigma);

        const size_t size = fan_in * fan_out;
        Eigen::Map<ArrayXt> arr(data, size);
        for(Eigen::Index i = 0; i < arr.size(); ++i){
            arr(i) = static_cast<f_type>(dist(rng));
        }
    }

    // Xavier/Glorot variance-scaled initialization. Fills data[0..fan_in*fan_out)
    // with i.i.d. N(0, (gain * sqrt(2 / (fan_in + fan_out)))^2) samples, where
    // `gain` encodes the layer's activation function (5/3 for tanh, 1 for
    // sigmoid/linear). Unlike Kaiming, Glorot scales by the harmonic mean of
    // fan_in and fan_out, which is derived to preserve the variance of both
    // forward activations and backward gradients. This is the initializer of
    // choice for tanh/sigmoid/linear layers.
    void eigen_initXavier(f_type* data, const size_t& fan_in, const size_t& fan_out,
                          const f_type& gain, const uint64_t& seed){
        std::mt19937_64 rng(seed);
        const double sigma = static_cast<double>(gain)
                             * std::sqrt(2.0 / (static_cast<double>(fan_in) + static_cast<double>(fan_out)));
        std::normal_distribution<double> dist(0.0, sigma);

        const size_t size = fan_in * fan_out;
        Eigen::Map<ArrayXt> arr(data, size);
        for(Eigen::Index i = 0; i < arr.size(); ++i){
            arr(i) = static_cast<f_type>(dist(rng));
        }
    }

}
