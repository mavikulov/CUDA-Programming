#pragma once

#include <cuda_fp16.h>

#include <cuda_helpers.h>

enum class MatrixLayout { RowMajor, ColMajor };

struct DeviceMatrix {
    __half* data;
    size_t rows;
    size_t cols;
    size_t stride;  // Distance in elements between first values of consecutive rows/columns
    MatrixLayout layout;
};

__device__ size_t GetIndex(const DeviceMatrix& m, size_t row, size_t col) {
    return m.layout == MatrixLayout::RowMajor ? row * m.stride + col : col * m.stride + row;
}

__global__ void KernelGEMM(DeviceMatrix a, DeviceMatrix b, DeviceMatrix c, DeviceMatrix d,
                           float alpha, float beta) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row >= d.rows) || (col >= d.cols)) {
        return;
    }

    float sum_ab = 0.0f;

    for (size_t k = 0; k < a.cols; ++k) {
        __half av = a.data[GetIndex(a, row, k)];
        __half bv = b.data[GetIndex(b, k, col)];
        __half prod = av * bv;
        sum_ab += static_cast<float>(prod);
    }

    float cv = c.data[GetIndex(c, row, col)];
    float result = alpha * sum_ab + beta * cv;
    d.data[GetIndex(d, row, col)] = static_cast<half>(result);
}

void GEMM(const DeviceMatrix& a, const DeviceMatrix& b, const DeviceMatrix& c, DeviceMatrix& d,
          float alpha, float beta) {
    dim3 blockSize(16, 16);
    dim3 gridSize((d.cols + blockSize.x - 1) / blockSize.x,
                  (d.rows + blockSize.y - 1) / blockSize.y);

    KernelGEMM<<<gridSize, blockSize>>>(a, b, c, d, alpha, beta);
}
