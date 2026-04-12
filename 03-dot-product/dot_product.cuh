#pragma once
#include <cuda_helpers.h>

size_t EstimateDotProductWorkspaceSizeBytes(size_t num_elements) {
    const int threads_per_block = 256;
    const int elements_per_thread = 2;
    const int elements_per_block = threads_per_block * elements_per_thread;
    const int reduce_block_size = 256;

    size_t num_blocks = (num_elements + elements_per_block - 1) / elements_per_block;
    size_t workspace_size = num_blocks * sizeof(float);
    if (num_blocks > reduce_block_size) {
        size_t num_blocks_l2 = (num_blocks + reduce_block_size - 1) / reduce_block_size;
        workspace_size += num_blocks_l2 * sizeof(float);
    }
    return workspace_size;
}

__global__ void KernelDotProductPartial(const float* lhs, const float* rhs, size_t num_elements,
                                        float* workspace) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int base = blockIdx.x * blockDim.x * 2;
    int idx0 = base + tid;
    int idx1 = base + tid + blockDim.x;

    float prod = 0.0f;
    if (idx0 < num_elements) {
        prod = lhs[idx0] * rhs[idx0];
    }

    if (idx1 < num_elements) {
        prod += lhs[idx1] * rhs[idx1];
    }

    temp[tid] = prod;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            temp[tid] += temp[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        workspace[blockIdx.x] = temp[0];
    }
}

__global__ void KernelDotProductReduce(float* workspace, size_t num_elements, float* out) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int base = blockIdx.x * blockDim.x * 2;
    int idx0 = base + tid;
    int idx1 = base + tid + blockDim.x;

    float sum = 0.0f;
    if (idx0 < num_elements) {
        sum = workspace[idx0];
    }

    if (idx1 < num_elements) {
        sum += workspace[idx1];
    }

    temp[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            temp[tid] += temp[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (gridDim.x == 1) {
            *out = temp[0];
        } else {
            workspace[num_elements + blockIdx.x] = temp[0];
        }
    }
}

void DotProduct(const float* lhs_device, const float* rhs_device, size_t num_elements,
                float* workspace_device, float* out_device) {
    const int blockSize = 256;
    const int blockSizeReduce = 256;
    const int elementsPerBlock = blockSize * 2;

    size_t dataBlocks = (num_elements + elementsPerBlock - 1) / elementsPerBlock;
    size_t sharedMemSize = blockSize * sizeof(float);

    KernelDotProductPartial<<<dataBlocks, blockSize, sharedMemSize>>>(
        lhs_device, rhs_device, num_elements, workspace_device);

    if (dataBlocks <= blockSizeReduce) {
        KernelDotProductReduce<<<1, blockSize, sharedMemSize>>>(workspace_device, dataBlocks,
                                                                out_device);
    } else {
        size_t partialSumBlocks = (dataBlocks + elementsPerBlock - 1) / elementsPerBlock;

        KernelDotProductReduce<<<partialSumBlocks, blockSize, sharedMemSize>>>(
            workspace_device, dataBlocks, workspace_device + dataBlocks);

        KernelDotProductReduce<<<1, blockSize, sharedMemSize>>>(workspace_device + dataBlocks,
                                                                partialSumBlocks, out_device);
    }
}
