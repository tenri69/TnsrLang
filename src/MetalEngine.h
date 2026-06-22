#pragma once
#include <vector>
#include <cstddef>

namespace tnsr {
    
    // Initializes the Metal GPU Device and Command Queue. 
    // Returns true if successful, false if no compatible GPU is found.
    bool MetalInitialize();
    
    // Checks if the Metal GPU is available and initialized.
    bool IsMetalAvailable();

    // Performs Matrix Multiplication C = A * B on the Apple Silicon GPU using Metal Performance Shaders.
    // A is (M x K) or (K x M) depending on transA.
    // B is (K x N) or (N x K) depending on transB.
    // C is (M x N).
    bool MetalComputeDotProduct(const std::vector<double>& A, 
                                const std::vector<double>& B, 
                                std::vector<double>& C, 
                                size_t M, size_t N, size_t K, 
                                bool transA, bool transB);
}
