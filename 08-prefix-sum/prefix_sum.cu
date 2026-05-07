#include "prefix_sum.cuh"

#include <cuda_runtime.h>

constexpr int kBlockSize = 1024;
constexpr int kElementsPerBlock = 2 * kBlockSize;

__global__ void BlockScanKernel(const int* input,
                                int* output,
                                int* block_sums,
                                size_t num_elements) {
    __shared__ int data[kElementsPerBlock];

    const int tid = threadIdx.x;

    const size_t block_start =
        static_cast<size_t>(blockIdx.x) * kElementsPerBlock;

    const size_t i0 = block_start + tid;
    const size_t i1 = block_start + tid + kBlockSize;

    data[tid] = (i0 < num_elements) ? input[i0] : 0;

    data[tid + kBlockSize] = (i1 < num_elements) ? input[i1] : 0;

    for (int stride = 1; stride < kElementsPerBlock; stride <<= 1) {
        __syncthreads();
        const int index = (tid + 1) * stride * 2 - 1;

        if (index < kElementsPerBlock) {
            data[index] += data[index - stride];
        }
    }

    if (tid == 0) {
        if (block_sums != nullptr) {
            block_sums[blockIdx.x] = data[kElementsPerBlock - 1];
        }

        data[kElementsPerBlock - 1] = 0;
    }

    for (int stride = kElementsPerBlock >> 1; stride >= 1; stride >>= 1) {
        __syncthreads();

        const int index = (tid + 1) * stride * 2 - 1;

        if (index < kElementsPerBlock) {
            const int temp = data[index - stride];
            data[index - stride] = data[index];
            data[index] += temp;
        }
    }

    __syncthreads();

    if (i0 < num_elements) {
        output[i0] = data[tid];
    }

    if (i1 < num_elements) {
        output[i1] = data[tid + kBlockSize];
    }
}

__global__ void AddBlockOffsetsKernel(int* output,
                                      const int* scanned_block_sums,
                                      size_t num_elements) {
    const size_t block_start =
        static_cast<size_t>(blockIdx.x) * kElementsPerBlock;

    const size_t i0 = block_start + threadIdx.x;
    const size_t i1 = i0 + kBlockSize;

    const int offset = scanned_block_sums[blockIdx.x];

    if (i0 < num_elements) {
        output[i0] += offset;
    }

    if (i1 < num_elements) {
        output[i1] += offset;
    }
}

__global__ void ExclusiveToInclusiveKernel(const int* input,
                                           int* output,
                                           size_t num_elements) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        output[idx] += input[idx];
    }
}

void PrefixSumImpl(const int* input,
                   int* output,
                   int* workspace,
                   size_t num_elements) {
    if (num_elements == 0) {
        return;
    }

    const size_t num_blocks =
        (num_elements + kElementsPerBlock - 1) / kElementsPerBlock;

    if (num_blocks == 1) {
        BlockScanKernel<<<1, kBlockSize>>>(
            input,
            output,
            nullptr,
            num_elements);

        return;
    }

    int* block_sums = workspace;
    int* scanned_block_sums = workspace + num_blocks;

    BlockScanKernel<<<num_blocks, kBlockSize>>>(
        input,
        output,
        block_sums,
        num_elements);

    PrefixSumImpl(
        block_sums,
        scanned_block_sums,
        nullptr,
        num_blocks);

    AddBlockOffsetsKernel<<<num_blocks, kBlockSize>>>(
        output,
        scanned_block_sums,
        num_elements);
}

__global__ void SetZeroKernel(int* output) {
    output[0] = 0;
}

size_t EstimatePrefixSumWorkspaceSizeBytes(size_t num_elements) {
    if (num_elements <= kElementsPerBlock) {
        return 0;
    }

    const size_t num_blocks =
        (num_elements + kElementsPerBlock - 1) / kElementsPerBlock;

    return 2 * num_blocks * sizeof(int);
}

void PrefixSumDevice(const int* input,
                     int* output,
                     int* workspace,
                     size_t num_elements) {
    SetZeroKernel<<<1, 1>>>(output);

    PrefixSumImpl(
        input,
        output + 1,
        workspace,
        num_elements);

    const int threads = 256;
    const int blocks =
        static_cast<int>((num_elements + threads - 1) / threads);

    ExclusiveToInclusiveKernel<<<blocks, threads>>>(
        input,
        output + 1,
        num_elements);
}