#include <limits>
#include "moe_topk.cuh"

// MoE router: for each token take Top-K experts by probability.
//
// logits:        [batchSize, numExperts], row-major, stride in elements (inputStride)
// outIdxs:  [batchSize, topK], int32 expert ids, stride idxsStride
// topkWeights:  [batchSize, topK], __half, stride outStride
//
// Tie-breaking when probabilities are equal: smaller expert index is preferred.

__device__ __forceinline__ void WarpReduceMax(float& val, int& idx) {
    for (int stride = 16; stride > 0; stride >>= 1) {
        float v = __shfl_down_sync(0xffffffff, val, stride);
        int i = __shfl_down_sync(0xffffffff, idx, stride);
        if ((v > val) || (v == val && i < idx)) {
            val = v;
            idx = i;
        }
    }
}

template <int K>
__global__ void MoeTopKKernel(size_t batchSize, size_t numExperts, size_t topK,
                              const __half* __restrict__ logits, size_t inputStride,
                              int32_t* __restrict__ outIdxs, size_t idxsStride,
                              __half* __restrict__ topkWeights, size_t outStride) {
    constexpr int warpSize = 32;
    int tid = threadIdx.x;
    int lane = tid & (warpSize - 1);
    int warp = tid >> 5;
    int warpsPerBlock = blockDim.x >> 5;
    int row = blockIdx.x * warpsPerBlock + warp;

    if (row >= batchSize) {
        return;
    }

    const __half* rowPtr = logits + row * inputStride;
    float vals[8];
    int idxs[8];

#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int idx = lane + j * warpSize;
        idxs[j] = idx;
        vals[j] =
            (idx < numExperts) ? __half2float(rowPtr[idx]) : std::numeric_limits<float>::lowest();
    }

#pragma unroll
    for (int k = 0; k < K; ++k) {
        if (k >= topK) {
            break;
        }

        float maxValue = std::numeric_limits<float>::lowest();
        int maxIdx = -1;

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int idx = idxs[j];
            float v = vals[j];
            if (idx < numExperts && ((v > maxValue) || (v == maxValue && idx < maxIdx))) {
                maxValue = v;
                maxIdx = idx;
            }
        }

        WarpReduceMax(maxValue, maxIdx);
        maxValue = __shfl_sync(0xffffffff, maxValue, 0);
        maxIdx = __shfl_sync(0xffffffff, maxIdx, 0);

        if (lane == 0) {
            outIdxs[row * idxsStride + k] = maxIdx;
            topkWeights[row * outStride + k] = __float2half(maxValue);
        }

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (idxs[j] == maxIdx) {
                vals[j] = std::numeric_limits<float>::lowest();
            }
        }
    }
}

void MoeTopK(size_t batchSize, size_t numExperts, size_t topK, const __half* logits,
             size_t inputStride, int32_t* outIdxs, size_t idxsStride, __half* topkWeights,
             size_t outStride) {
    int blockSize = 128;
    int gridSize = (batchSize + (blockSize / 32) - 1) / (blockSize / 32);
    switch (topK) {
        case 1:
            MoeTopKKernel<1><<<gridSize, blockSize>>>(batchSize, numExperts, topK, logits,
                                                      inputStride, outIdxs, idxsStride, topkWeights,
                                                      outStride);
            break;
        case 2:
            MoeTopKKernel<2><<<gridSize, blockSize>>>(batchSize, numExperts, topK, logits,
                                                      inputStride, outIdxs, idxsStride, topkWeights,
                                                      outStride);
            break;
        case 3:
            MoeTopKKernel<3><<<gridSize, blockSize>>>(batchSize, numExperts, topK, logits,
                                                      inputStride, outIdxs, idxsStride, topkWeights,
                                                      outStride);
            break;
        case 4:
            MoeTopKKernel<4><<<gridSize, blockSize>>>(batchSize, numExperts, topK, logits,
                                                      inputStride, outIdxs, idxsStride, topkWeights,
                                                      outStride);
            break;
        case 5:
            MoeTopKKernel<5><<<gridSize, blockSize>>>(batchSize, numExperts, topK, logits,
                                                      inputStride, outIdxs, idxsStride, topkWeights,
                                                      outStride);
            break;
        case 6:
            MoeTopKKernel<6><<<gridSize, blockSize>>>(batchSize, numExperts, topK, logits,
                                                      inputStride, outIdxs, idxsStride, topkWeights,
                                                      outStride);
            break;
        case 7:
            MoeTopKKernel<7><<<gridSize, blockSize>>>(batchSize, numExperts, topK, logits,
                                                      inputStride, outIdxs, idxsStride, topkWeights,
                                                      outStride);
            break;
        default:
            MoeTopKKernel<8><<<gridSize, blockSize>>>(batchSize, numExperts, topK, logits,
                                                      inputStride, outIdxs, idxsStride, topkWeights,
                                                      outStride);
            break;
    }
}
