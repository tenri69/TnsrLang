#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "MetalEngine.h"
#include <iostream>

namespace tnsr {

    static id<MTLDevice> device = nil;
    static id<MTLCommandQueue> commandQueue = nil;
    static bool is_initialized = false;

    bool MetalInitialize() {
        if (is_initialized) return true;

        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "[Metal] Failed to find a compatible Metal device.\n";
            return false;
        }

        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            std::cerr << "[Metal] Failed to create command queue.\n";
            return false;
        }

        is_initialized = true;
        std::cout << "[Metal] Successfully initialized GPU: " << [[device name] UTF8String] << "\n";
        return true;
    }

    bool IsMetalAvailable() {
        return is_initialized;
    }

    bool MetalComputeDotProduct(const std::vector<double>& A, 
                                const std::vector<double>& B, 
                                std::vector<double>& C, 
                                size_t M, size_t N, size_t K, 
                                bool transA, bool transB) {
        if (!is_initialized) return false;

        // MPS only supports Float32. We must cast our Double vectors to Float.
        std::vector<float> A_f32(A.size());
        std::vector<float> B_f32(B.size());
        
        for (size_t i = 0; i < A.size(); ++i) A_f32[i] = (float)A[i];
        for (size_t i = 0; i < B.size(); ++i) B_f32[i] = (float)B[i];

        size_t rowBytesA = transA ? M * sizeof(float) : K * sizeof(float);
        size_t rowBytesB = transB ? K * sizeof(float) : N * sizeof(float);
        size_t rowBytesC = N * sizeof(float);

        id<MTLBuffer> bufferA = [device newBufferWithBytes:A_f32.data() length:A_f32.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:B_f32.data() length:B_f32.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithLength:M * N * sizeof(float) options:MTLResourceStorageModeShared];

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:(transA ? K : M) columns:(transA ? M : K) rowBytes:rowBytesA dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:(transB ? N : K) columns:(transB ? K : N) rowBytes:rowBytesB dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:rowBytesC dataType:MPSDataTypeFloat32];

        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        MPSMatrixMultiplication *mul = [[MPSMatrixMultiplication alloc] initWithDevice:device 
                                                                             transposeLeft:transA 
                                                                            transposeRight:transB 
                                                                            resultRows:M 
                                                                         resultColumns:N 
                                                                       interiorColumns:K 
                                                                                 alpha:1.0 
                                                                                  beta:0.0];

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [mul encodeToCommandBuffer:commandBuffer leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        float* result_ptr = (float*)[bufferC contents];
        for (size_t i = 0; i < M * N; ++i) {
            C[i] = (double)result_ptr[i];
        }

        return true;
    }
}
