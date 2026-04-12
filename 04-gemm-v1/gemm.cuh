#pragma once

#include <cassert>
#include <cuda_fp16.h>
#include <cuda_helpers.h>

enum class MatrixLayout { RowMajor, ColMajor };

struct DeviceMatrix {
    __half* data;
    size_t rows;
    size_t cols;
    size_t stride;
    MatrixLayout layout;
};

constexpr int TILE = 64;
constexpr int BLOCK_ROWS = 4;

__global__ void KernelGEMM(DeviceMatrix a, DeviceMatrix b, DeviceMatrix c, DeviceMatrix d,
                           float alpha, float beta) {
    __shared__ __half As[TILE][TILE + 1];
    __shared__ __half Bs[TILE][TILE + 1];
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;

    float accumulator[TILE / BLOCK_ROWS] = {0.f};

    for (int tile_idx = 0; tile_idx < (a.cols + TILE - 1) / TILE; ++tile_idx) {
#pragma unroll
        for (int i = 0; i < TILE; i += BLOCK_ROWS) {
            int r = row + i;
            int c = tile_idx * TILE + threadIdx.x;

            if (r < a.rows && c < a.cols) {
                As[threadIdx.y + i][threadIdx.x] = a.data[r * a.stride + c];
            } else {
                As[threadIdx.y + i][threadIdx.x] = static_cast<half>(0.0f);
            }
        }

#pragma unroll
        for (int i = 0; i < TILE; i += BLOCK_ROWS) {
            int local_n = threadIdx.y + i;
            int local_k = threadIdx.x;
            int r = tile_idx * TILE + local_k;
            int c = blockIdx.x * TILE + local_n;

            if (r < b.rows && c < b.cols) {
                Bs[local_n][local_k] = b.data[c * b.stride + r];
            } else {
                Bs[local_n][local_k] = static_cast<half>(0.0f);
            }
        }

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            __half bv = Bs[threadIdx.x][k];

#pragma unroll
            for (int i = 0; i < TILE; i += BLOCK_ROWS) {
                int acc_idx = i / BLOCK_ROWS;
                __half av = As[threadIdx.y + i][k];
                __half prod = av * bv;
                accumulator[acc_idx] += static_cast<float>(prod);
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TILE; i += BLOCK_ROWS) {
        int r = row + i;
        int acc_idx = i / BLOCK_ROWS;
        if (r < d.rows && col < d.cols) {
            float cv = static_cast<float>(c.data[col * c.stride + r]);
            float result = alpha * accumulator[acc_idx] + beta * cv;
            d.data[col * c.stride + r] = static_cast<half>(result);
        }
    }
}

void GEMM(const DeviceMatrix& a, const DeviceMatrix& b, const DeviceMatrix& c, DeviceMatrix& d,
          float alpha, float beta) {
    assert(a.layout == MatrixLayout::RowMajor);
    assert(b.layout == MatrixLayout::ColMajor);
    assert(c.layout == MatrixLayout::ColMajor);
    assert(d.layout == MatrixLayout::ColMajor);

    dim3 block(TILE, BLOCK_ROWS);
    dim3 grid((d.cols + TILE - 1) / TILE, (d.rows + TILE - 1) / TILE);

    KernelGEMM<<<grid, block>>>(a, b, c, d, alpha, beta);
}
