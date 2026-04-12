#pragma once

#include <vector>

#include <cuda_helpers.h>

float* AllocDeviceVector(size_t num_elements) {
    float* vecDevice = nullptr;
    CheckStatus(cudaMalloc(&vecDevice, num_elements * sizeof(float)));
    return vecDevice;
}

void FreeDeviceVector(float* device_ptr) {
    CheckStatus(cudaFree(device_ptr));
}

void CopyHostVectorToDevice(const std::vector<float>& vector_host, float* dst_device_ptr) {
    CheckStatus(cudaMemcpy(dst_device_ptr, vector_host.data(), vector_host.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
}

std::vector<float> CopyDeviceVectorToHost(const float* ptr_device, size_t num_elements) {
    std::vector<float> hostVector(num_elements);
    CheckStatus(cudaMemcpy(hostVector.data(), ptr_device, num_elements * sizeof(float),
                           cudaMemcpyDeviceToHost));
    return hostVector;
}

__global__ void AddVectorsKernel(const float* a, const float* b, float* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

void AddDeviceVectors(const float* left_device, const float* right_device, float* out_device,
                      size_t num_elements) {
    size_t blockSize = 1024;
    size_t gridSize = (num_elements + (blockSize - 1)) / blockSize;
    AddVectorsKernel<<<gridSize, blockSize>>>(left_device, right_device, out_device, num_elements);
    CheckStatus(cudaDeviceSynchronize());
}
