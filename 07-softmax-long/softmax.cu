#include "softmax.cuh"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

// Using approaches from https://github.com/SzymonOzog/FastSoftmax/blob/main/kernels.cu

__inline__ __device__ float WarpReduceMax(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__inline__ __device__ float WarpReduceSum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <int BlockThreads>
__inline__ __device__ float BlockReduceMax(float value) {
    __shared__ float shared[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = WarpReduceMax(value);

    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    float result = -FLT_MAX;

    if (warp == 0) {
        result = lane < ((BlockThreads + 31) / 32) ? shared[lane] : -FLT_MAX;
        result = WarpReduceMax(result);

        if (lane == 0) {
            shared[0] = result;
        }
    }

    __syncthreads();

    return shared[0];
}

template <int BlockThreads>
__inline__ __device__ float BlockReduceSum(float value) {
    __shared__ float shared[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = WarpReduceSum(value);

    if (lane == 0) {
        shared[warp] = value;
    }

    __syncthreads();

    float result = 0.0f;

    if (warp == 0) {
        result = lane < ((BlockThreads + 31) / 32) ? shared[lane] : 0.0f;
        result = WarpReduceSum(result);

        if (lane == 0) {
            shared[0] = result;
        }
    }

    __syncthreads();

    return shared[0];
}

template <int BlockThreads>
__global__ void LongSoftmaxScalarKernel(size_t rows, size_t cols, const float* __restrict__ input,
                                        size_t input_stride, float* __restrict__ output,
                                        size_t output_stride) {
    const size_t row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    const float* row_input = input + row * input_stride;
    float* row_output = output + row * output_stride;
    float local_max = -FLT_MAX;

    for (size_t col = threadIdx.x; col < cols; col += BlockThreads) {
        local_max = fmaxf(local_max, row_input[col]);
    }

    const float row_max = BlockReduceMax<BlockThreads>(local_max);
    float local_sum = 0.0f;

    for (size_t col = threadIdx.x; col < cols; col += BlockThreads) {
        local_sum += expf(row_input[col] - row_max);
    }

    const float row_sum = BlockReduceSum<BlockThreads>(local_sum);
    const float inv_sum = 1.0f / row_sum;

    for (size_t col = threadIdx.x; col < cols; col += BlockThreads) {
        row_output[col] = expf(row_input[col] - row_max) * inv_sum;
    }
}

template <int BlockThreads>
__global__ void LongSoftmaxVec4Kernel(size_t rows, size_t cols, const float* __restrict__ input,
                                      size_t input_stride, float* __restrict__ output,
                                      size_t output_stride) {
    const size_t row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    const size_t cols4 = cols / 4;
    const float4* row_input4 = reinterpret_cast<const float4*>(input + row * input_stride);
    float4* row_output4 = reinterpret_cast<float4*>(output + row * output_stride);
    float local_max = -FLT_MAX;

    for (size_t i = threadIdx.x; i < cols4; i += BlockThreads) {
        const float4 v = row_input4[i];
        local_max = fmaxf(local_max, v.x);
        local_max = fmaxf(local_max, v.y);
        local_max = fmaxf(local_max, v.z);
        local_max = fmaxf(local_max, v.w);
    }

    const float row_max = BlockReduceMax<BlockThreads>(local_max);
    float local_sum = 0.0f;

    for (size_t i = threadIdx.x; i < cols4; i += BlockThreads) {
        const float4 v = row_input4[i];
        local_sum += expf(v.x - row_max);
        local_sum += expf(v.y - row_max);
        local_sum += expf(v.z - row_max);
        local_sum += expf(v.w - row_max);
    }

    const float row_sum = BlockReduceSum<BlockThreads>(local_sum);
    const float inv_sum = 1.0f / row_sum;

    for (size_t i = threadIdx.x; i < cols4; i += BlockThreads) {
        float4 v = row_input4[i];
        v.x = expf(v.x - row_max) * inv_sum;
        v.y = expf(v.y - row_max) * inv_sum;
        v.z = expf(v.z - row_max) * inv_sum;
        v.w = expf(v.w - row_max) * inv_sum;
        row_output4[i] = v;
    }
}

void Softmax(size_t rows, size_t cols, const float* d_input_matrix, size_t input_stride,
             float* d_out, size_t out_stride, cudaStream_t stream) {
    if (rows == 0 || cols == 0) {
        return;
    }
    const int blockSize = 256;
    const size_t gridSize = rows;
    const bool use_vec4 = (cols % 4 == 0 && input_stride % 4 == 0 && out_stride % 4 == 0 &&
                           reinterpret_cast<std::uintptr_t>(d_input_matrix) % 16 == 0 &&
                           reinterpret_cast<std::uintptr_t>(d_out) % 16 == 0);

    if (use_vec4) {
        LongSoftmaxVec4Kernel<blockSize><<<gridSize, blockSize, 0, stream>>>(
            rows, cols, d_input_matrix, input_stride, d_out, out_stride);
    } else {
        LongSoftmaxScalarKernel<blockSize><<<gridSize, blockSize, 0, stream>>>(
            rows, cols, d_input_matrix, input_stride, d_out, out_stride);
    }
}
