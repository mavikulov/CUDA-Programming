#include "quaternions.cuh"

#include <cassert>

#include <cuda_runtime.h>

__global__ void QuaternionsReduceKernel(size_t rows, size_t cols,
                                        const Quaternion* __restrict__ inp, size_t inp_stride,
                                        Quaternion* __restrict__ out) {
    size_t tid = threadIdx.x;
    const int blockSize = 256;
    const size_t row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    const size_t chunk_size = (cols + blockSize - 1) / blockSize;
    const size_t start = tid * chunk_size;
    const size_t end =
        (start < cols) ? ((start + chunk_size < cols) ? start + chunk_size : cols) : start;

    extern __shared__ Quaternion temp[];
    QuaternionMultiplier mul;
    Quaternion local{1.f, 0.f, 0.f, 0.f};

#pragma unroll
    for (size_t c = start; c < end; ++c) {
        local = mul(local, inp[row * inp_stride + c]);
    }

    temp[tid] = local;
    __syncthreads();

    if (tid == 0) {
        Quaternion result{1.f, 0.f, 0.f, 0.f};
        for (size_t t = 0; t < blockSize; ++t) {
            const size_t chunk_start = t * chunk_size;
            if (chunk_start < cols) {
                result = mul(result, temp[t]);
            }
        }
        out[row] = result;
    }
}

void QuaternionsReduce(size_t rows, size_t cols, const Quaternion* inp, size_t inp_stride,
                       Quaternion* out, cudaStream_t stream) {
    constexpr int blockSize = 256;
    const size_t sharedMemSize = blockSize * sizeof(Quaternion);

    QuaternionsReduceKernel<<<rows, blockSize, sharedMemSize, stream>>>(rows, cols, inp, inp_stride,
                                                                        out);
}
