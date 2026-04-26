#include "moe_topk_hist.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

// Same as MoeTopK plus expertHistogram[e] = count of assignments to expert e.
// Zero expertHistogram (length numExperts) before calling.

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

template <int K, bool UseSharedHist>
__global__ void MoeTopKHistKernel(size_t batchSize, size_t numExperts,
                                  const __half* __restrict__ logits, size_t inputStride,
                                  int32_t* __restrict__ outIdxs, size_t idxsStride,
                                  __half* __restrict__ topkWeights, size_t outStride,
                                  unsigned int* __restrict__ expertHistogram) {
    extern __shared__ unsigned int temp[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    if constexpr (UseSharedHist) {
        for (size_t expert = threadIdx.x; expert < numExperts; expert += blockDim.x) {
            temp[expert] = 0;
        }
        __syncthreads();
    }

    const size_t row = static_cast<size_t>(blockIdx.x) * 4 + warp;

    if (row < batchSize) {
        const __half* rowLogits = logits + row * inputStride;
        int selectedIdx[8];

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            selectedIdx[i] = -1;
        }

#pragma unroll
        for (int k = 0; k < K; ++k) {
            float bestValue = -FLT_MAX;
            int bestIdx = static_cast<int>(numExperts);

            for (size_t expert = lane; expert < numExperts; expert += 32) {
                const int expertIdx = static_cast<int>(expert);
                bool isSelected = false;

#pragma unroll
                for (int prev = 0; prev < K; ++prev) {
                    if (prev < k && selectedIdx[prev] == expertIdx) {
                        isSelected = true;
                    }
                }

                if (!isSelected) {
                    const float value = __half2float(rowLogits[expert]);

                    if (value > bestValue || (value == bestValue && expertIdx < bestIdx)) {
                        bestValue = value;
                        bestIdx = expertIdx;
                    }
                }
            }

            WarpReduceMax(bestValue, bestIdx);

            const int selected = __shfl_sync(0xffffffff, bestIdx, 0);
            const float selectedValue = __shfl_sync(0xffffffff, bestValue, 0);

            selectedIdx[k] = selected;

            if (lane == 0) {
                outIdxs[row * idxsStride + k] = selected;
                topkWeights[row * outStride + k] = __float2half(selectedValue);

                if constexpr (UseSharedHist) {
                    atomicAdd(&temp[selected], 1u);
                } else {
                    atomicAdd(&expertHistogram[selected], 1u);
                }
            }
        }
    }

    if constexpr (UseSharedHist) {
        __syncthreads();

        for (size_t expert = threadIdx.x; expert < numExperts; expert += blockDim.x) {
            const unsigned int count = temp[expert];
            if (count != 0) {
                atomicAdd(&expertHistogram[expert], count);
            }
        }
    }
}

template <bool UseSharedHist>
__global__ void MoeTop1HistKernel(size_t batchSize, size_t numExperts,
                                  const __half* __restrict__ logits, size_t inputStride,
                                  int32_t* __restrict__ outIdxs, size_t idxsStride,
                                  __half* __restrict__ topkWeights, size_t outStride,
                                  unsigned int* __restrict__ expertHistogram) {
    extern __shared__ unsigned int temp[];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    if constexpr (UseSharedHist) {
        for (size_t expert = threadIdx.x; expert < numExperts; expert += blockDim.x) {
            temp[expert] = 0;
        }
        __syncthreads();
    }

    const size_t row = static_cast<size_t>(blockIdx.x) * 4 + warp;

    if (row < batchSize) {
        const __half* rowLogits = logits + row * inputStride;

        float bestValue = -FLT_MAX;
        int bestIdx = static_cast<int>(numExperts);

        for (size_t expert = lane; expert < numExperts; expert += 32) {
            const int expertIdx = static_cast<int>(expert);
            const float value = static_cast<float>(rowLogits[expert]);

            if (value > bestValue || (value == bestValue && expertIdx < bestIdx)) {
                bestValue = value;
                bestIdx = expertIdx;
            }
        }

        WarpReduceMax(bestValue, bestIdx);

        if (lane == 0) {
            outIdxs[row * idxsStride] = bestIdx;
            topkWeights[row * outStride] = static_cast<half>(bestValue);

            if constexpr (UseSharedHist) {
                atomicAdd(&temp[bestIdx], 1u);
            } else {
                atomicAdd(&expertHistogram[bestIdx], 1u);
            }
        }
    }

    if constexpr (UseSharedHist) {
        __syncthreads();

        for (size_t expert = threadIdx.x; expert < numExperts; expert += blockDim.x) {
            unsigned int count = temp[expert];
            if (count != 0) {
                atomicAdd(&expertHistogram[expert], count);
            }
        }
    }
}

template <bool UseSharedHist>
void LaunchMoeTopKHistKernel(size_t batchSize, size_t numExperts, size_t topK, const __half* logits,
                             size_t inputStride, int32_t* outIdxs, size_t idxsStride,
                             __half* topkWeights, size_t outStride, unsigned int* expertHistogram) {
    const int warpSize = 32;
    const int blockSize = 128;
    const int warpsPerBlock = blockSize / warpSize;
    const int gridSize = (batchSize + warpsPerBlock - 1) / warpsPerBlock;
    const size_t sharedMemSize = UseSharedHist ? numExperts * sizeof(unsigned int) : 0;

    switch (topK) {
        case 1:
            MoeTop1HistKernel<UseSharedHist><<<gridSize, blockSize, sharedMemSize>>>(
                batchSize, numExperts, logits, inputStride, outIdxs, idxsStride, topkWeights,
                outStride, expertHistogram);
            break;

        case 2:
            MoeTopKHistKernel<2, UseSharedHist><<<gridSize, blockSize, sharedMemSize>>>(
                batchSize, numExperts, logits, inputStride, outIdxs, idxsStride, topkWeights,
                outStride, expertHistogram);
            break;

        case 3:
            MoeTopKHistKernel<3, UseSharedHist><<<gridSize, blockSize, sharedMemSize>>>(
                batchSize, numExperts, logits, inputStride, outIdxs, idxsStride, topkWeights,
                outStride, expertHistogram);
            break;

        case 4:
            MoeTopKHistKernel<4, UseSharedHist><<<gridSize, blockSize, sharedMemSize>>>(
                batchSize, numExperts, logits, inputStride, outIdxs, idxsStride, topkWeights,
                outStride, expertHistogram);
            break;

        case 5:
            MoeTopKHistKernel<5, UseSharedHist><<<gridSize, blockSize, sharedMemSize>>>(
                batchSize, numExperts, logits, inputStride, outIdxs, idxsStride, topkWeights,
                outStride, expertHistogram);
            break;

        case 6:
            MoeTopKHistKernel<6, UseSharedHist><<<gridSize, blockSize, sharedMemSize>>>(
                batchSize, numExperts, logits, inputStride, outIdxs, idxsStride, topkWeights,
                outStride, expertHistogram);
            break;

        case 7:
            MoeTopKHistKernel<7, UseSharedHist><<<gridSize, blockSize, sharedMemSize>>>(
                batchSize, numExperts, logits, inputStride, outIdxs, idxsStride, topkWeights,
                outStride, expertHistogram);
            break;

        default:
            MoeTopKHistKernel<8, UseSharedHist><<<gridSize, blockSize, sharedMemSize>>>(
                batchSize, numExperts, logits, inputStride, outIdxs, idxsStride, topkWeights,
                outStride, expertHistogram);
            break;
    }
}

void MoeTopKHist(size_t batchSize, size_t numExperts, size_t topK, const __half* logits,
                 size_t inputStride, int32_t* outIdxs, size_t idxsStride, __half* topkWeights,
                 size_t outStride, unsigned int* expertHistogram) {
    if (batchSize == 0 || numExperts == 0 || topK == 0) {
        return;
    }

    if (numExperts <= 4096 && batchSize >= 128) {
        LaunchMoeTopKHistKernel<true>(batchSize, numExperts, topK, logits, inputStride, outIdxs,
                                      idxsStride, topkWeights, outStride, expertHistogram);
    } else {
        LaunchMoeTopKHistKernel<false>(batchSize, numExperts, topK, logits, inputStride, outIdxs,
                                       idxsStride, topkWeights, outStride, expertHistogram);
    }
}
