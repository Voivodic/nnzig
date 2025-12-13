#include <Eigen/Dense>
#include <cmath>

// Declare the functions defined in the zig file
extern "C" {
}

// Define the functions that will be exported to zig
extern "C"{

    // None activation for half precision
    void noneHalf(half* df, size_t N){
        Eigen::Map<vectorh> v(df, N);
    
    }

}
