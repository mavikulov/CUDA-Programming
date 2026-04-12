#pragma once

__global__ void Map(const float* data, float* out) {
    int idx = threadIdx.x;
    out[idx] = data[idx] + 10.f;
}

__global__ void Zip(const float* left, const float* right, float* out) {
    int idx = threadIdx.x;
    out[idx] = left[idx] + right[idx];
}

__global__ void Guard(const float* data, float* out, size_t size) {
    int idx = threadIdx.x;
    if (idx < size) {
        out[idx] = data[idx] + 10.f;
    }
}

__global__ void Block(const float* data, float* out, float value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = data[idx] + value;
    }
}
