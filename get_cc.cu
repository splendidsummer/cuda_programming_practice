#include <iostream>
#include <cuda_runtime.h>
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        std::cerr << "No CUDA devices or CUDA not installed\n";
        return 1;
    }
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name
                  << "  compute capability = " << prop.major << "." << prop.minor << std::endl;
    }
    return 0;
}
