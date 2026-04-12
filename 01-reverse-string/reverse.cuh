#pragma once

#include <cstddef>

#include <cuda_helpers.h>

__global__ void KernelReverseString(char* str, size_t length) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < length / 2) {
        char tmp = str[idx];
        str[idx] = str[length - 1 - idx];
        str[length - 1 - idx] = tmp;
    }
}

void ReverseDeviceStringInplace(char* str, size_t length) {
    if (length > 1) {
        size_t numThreads = length / 2;
        size_t blockSize = 256;
        size_t gridSize = (numThreads + (blockSize - 1)) / blockSize;
        KernelReverseString<<<gridSize, blockSize>>>(str, length);
        CheckStatus(cudaDeviceSynchronize());
    }
}
