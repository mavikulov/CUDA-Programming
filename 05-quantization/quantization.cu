#include "quantization.cuh"

// https://medium.com/@rimikadhara/7-step-optimization-of-parallel-reduction-with-cuda-33a3b2feafd8
__device__ __forceinline__ float WardReduceMax(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__global__ void QuantizationKernel(size_t rows, size_t cols,
                                   const float* __restrict__ d_input_matrix,
                                   const float* __restrict__ d_balance_factors, size_t input_stride,
                                   size_t out_stride, int8_t* d_out, float* d_out_scales) {
    const float mScale = 1e-5f;
    const int row = blockIdx.x;
    size_t tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    extern __shared__ float temp[];
    float row_max = 0.0f;
    for (size_t c = tid; c < cols; c += blockDim.x) {
        float w = d_input_matrix[row * input_stride + c];
        float s = d_balance_factors[c];
        row_max = fmaxf(row_max, fabsf(w + s));
    }

    row_max = WardReduceMax(row_max);
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    if (lane == 0) {
        temp[warp_id] = row_max;
    }
    __syncthreads();

    float max_abs = 0.0f;
    if (warp_id == 0) {
        float value = (lane < num_warps) ? temp[lane] : 0.0f;
        value = WardReduceMax(value);
        max_abs = value;
    }
    __syncthreads();

    if (tid == 0) {
        float scale = 127.0f / fmaxf(max_abs, mScale);
        temp[0] = scale;
        d_out_scales[row] = scale;
    }
    __syncthreads();

    float scale = temp[0];

    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        size_t inIdx = row * input_stride + c;
        size_t outIdx = row * out_stride + c;
        float w = __ldg(&d_input_matrix[inIdx]);
        float s = __ldg(&d_balance_factors[c]);
        float q_val = (w + s) * scale;
        int32_t q_int = __float2int_rn(q_val);

        if (q_int > 127) {
            q_int = 127;
        }

        if (q_int < -128) {
            q_int = -128;
        }

        d_out[outIdx] = static_cast<int8_t>(q_int);
    }
}

void Quantization(size_t rows, size_t cols, const float* d_input_matrix,
                  const float* d_balance_factors, size_t input_stride, size_t out_stride,
                  int8_t* d_out, float* d_out_scales) {
    const int warpSize = 32;
    const int blockSize = 256;
    const int numWarps = (blockSize + warpSize - 1) >> 5;
    size_t sharedMemSize = numWarps * sizeof(float);

    QuantizationKernel<<<rows, blockSize, sharedMemSize>>>(rows, cols, d_input_matrix,
                                                           d_balance_factors, input_stride,
                                                           out_stride, d_out, d_out_scales);
}
