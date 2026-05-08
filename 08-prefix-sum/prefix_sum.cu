#include "prefix_sum.cuh"

#include <cuda_runtime.h>

constexpr int kWarpSize = 32;
constexpr int kBlockSize = 256;
constexpr int kItemsPerThread = 8;
constexpr int kElementsPerBlock = kBlockSize * kItemsPerThread;
constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;

__device__ __forceinline__ int WarpInclusiveScan(int val) {
    const unsigned int mask = 0xFFFFFFFF;
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const int n = __shfl_up_sync(mask, val, offset);
        if ((threadIdx.x & (kWarpSize - 1)) >= offset) {
            val += n;
        }
    }
    return val;
}

__device__ __forceinline__ int BlockExclusiveScan(int (&items)[kItemsPerThread]) {
    __shared__ int warp_sums[kWarpsPerBlock];
    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp_id = tid >> 5;

    int thread_sum = 0;
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_sum += items[i];
    }

    int thread_scan = WarpInclusiveScan(thread_sum);
    int warp_excl = thread_scan - thread_sum;

    if (lane == kWarpSize - 1) {
        warp_sums[warp_id] = thread_scan;
    }
    __syncthreads();

    if (warp_id == 0) {
        int v = (lane < kWarpsPerBlock) ? warp_sums[lane] : 0;
        v = WarpInclusiveScan(v);
        if (lane < kWarpsPerBlock) {
            warp_sums[lane] = v;
        }
    }
    __syncthreads();

    const int warp_prefix = (warp_id == 0) ? 0 : warp_sums[warp_id - 1];
    const int thread_prefix = warp_prefix + warp_excl;
    int running = thread_prefix;
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        const int v = items[i];
        items[i] = running;
        running += v;
    }

    return warp_sums[kWarpsPerBlock - 1];
}

__device__ __forceinline__ void LoadBlockedFullVec(const int* __restrict__ input,
                                                   size_t block_start,
                                                   int (&items)[kItemsPerThread]) {
    const int tid = threadIdx.x;
    const size_t base = block_start + static_cast<size_t>(tid) * kItemsPerThread;
    const int4* in4 = reinterpret_cast<const int4*>(input + base);

    const int4 a = in4[0];
    const int4 b = in4[1];
    items[0] = a.x;
    items[1] = a.y;
    items[2] = a.z;
    items[3] = a.w;
    items[4] = b.x;
    items[5] = b.y;
    items[6] = b.z;
    items[7] = b.w;
}

__device__ __forceinline__ void StoreBlockedFullVec(int* __restrict__ output, size_t block_start,
                                                    const int (&items)[kItemsPerThread]) {
    const int tid = threadIdx.x;
    const size_t base = block_start + static_cast<size_t>(tid) * kItemsPerThread;
    int4* out4 = reinterpret_cast<int4*>(output + base);

    out4[0] = make_int4(items[0], items[1], items[2], items[3]);
    out4[1] = make_int4(items[4], items[5], items[6], items[7]);
}

__device__ __forceinline__ void LoadBlockedTail(const int* __restrict__ input, size_t block_start,
                                                size_t num_elements,
                                                int (&items)[kItemsPerThread]) {
    const int tid = threadIdx.x;
    const size_t base = block_start + static_cast<size_t>(tid) * kItemsPerThread;
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        const size_t idx = base + i;
        items[i] = (idx < num_elements) ? input[idx] : 0;
    }
}

__device__ __forceinline__ void StoreBlockedTail(int* __restrict__ output, size_t block_start,
                                                 size_t num_elements,
                                                 const int (&items)[kItemsPerThread]) {
    const int tid = threadIdx.x;
    const size_t base = block_start + static_cast<size_t>(tid) * kItemsPerThread;
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        const size_t idx = base + i;
        if (idx < num_elements) {
            output[idx] = items[i];
        }
    }
}

__global__ void BlockExclusiveScanKernel(const int* __restrict__ input, int* __restrict__ output,
                                         int* __restrict__ block_sums, size_t num_elements) {
    const size_t block_start = static_cast<size_t>(blockIdx.x) * kElementsPerBlock;
    const bool full_block = (block_start + kElementsPerBlock <= num_elements);

    int items[kItemsPerThread];
    if (full_block) {
        LoadBlockedFullVec(input, block_start, items);
    } else {
        LoadBlockedTail(input, block_start, num_elements, items);
    }

    const int total = BlockExclusiveScan(items);

    if (full_block) {
        StoreBlockedFullVec(output, block_start, items);
    } else {
        StoreBlockedTail(output, block_start, num_elements, items);
    }

    if (block_sums != nullptr && threadIdx.x == 0) {
        block_sums[blockIdx.x] = total;
    }
}

__device__ __forceinline__ int BlockInclusiveScan(int (&items)[kItemsPerThread]) {
    __shared__ int warp_sums[kWarpsPerBlock];
    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp_id = tid >> 5;

#pragma unroll
    for (int i = 1; i < kItemsPerThread; ++i) {
        items[i] += items[i - 1];
    }

    const int thread_sum = items[kItemsPerThread - 1];
    const int thread_scan = WarpInclusiveScan(thread_sum);
    const int warp_excl = thread_scan - thread_sum;

    if (lane == kWarpSize - 1) {
        warp_sums[warp_id] = thread_scan;
    }
    __syncthreads();

    if (warp_id == 0) {
        int v = (lane < kWarpsPerBlock) ? warp_sums[lane] : 0;
        v = WarpInclusiveScan(v);
        if (lane < kWarpsPerBlock) {
            warp_sums[lane] = v;
        }
    }
    __syncthreads();

    const int warp_prefix = (warp_id == 0) ? 0 : warp_sums[warp_id - 1];
    const int thread_prefix = warp_prefix + warp_excl;

#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        items[i] += thread_prefix;
    }

    return warp_sums[kWarpsPerBlock - 1];
}

__global__ void BlockInclusiveScanKernel(const int* __restrict__ input, int* output,
                                         size_t num_elements) {
    int items[kItemsPerThread];
    LoadBlockedTail(input, 0, num_elements, items);
    BlockInclusiveScan(items);
    StoreBlockedTail(output, 0, num_elements, items);
}

__global__ void AddOffsetsKernel(int* __restrict__ output,
                                 const int* __restrict__ scanned_block_sums, size_t num_elements) {
    if (blockIdx.x == 0) {
        return;
    }
    const int offset = scanned_block_sums[blockIdx.x - 1];
    const size_t block_start = static_cast<size_t>(blockIdx.x) * kElementsPerBlock;
    const bool full_block = (block_start + kElementsPerBlock <= num_elements);
    const int tid = threadIdx.x;
    const size_t base = block_start + static_cast<size_t>(tid) * kItemsPerThread;

    if (full_block) {
        int4* p4 = reinterpret_cast<int4*>(output + base);
        int4 a = p4[0];
        int4 b = p4[1];
        a.x += offset;
        a.y += offset;
        a.z += offset;
        a.w += offset;
        b.x += offset;
        b.y += offset;
        b.z += offset;
        b.w += offset;
        p4[0] = a;
        p4[1] = b;
    } else {
#pragma unroll
        for (int i = 0; i < kItemsPerThread; ++i) {
            const size_t idx = base + i;
            if (idx < num_elements) {
                output[idx] += offset;
            }
        }
    }
}

__global__ void WriteLastElementKernel(int* __restrict__ output,
                                       const int* __restrict__ scanned_block_sums,
                                       size_t num_blocks, size_t num_elements) {
    output[num_elements] = scanned_block_sums[num_blocks - 1];
}

__global__ void SinglePassScanKernel(const int* __restrict__ input, int* __restrict__ output,
                                     size_t num_elements) {
    int items[kItemsPerThread];
    LoadBlockedTail(input, 0, num_elements, items);
    const int total = BlockExclusiveScan(items);
    StoreBlockedTail(output, 0, num_elements, items);

    if (threadIdx.x == 0) {
        output[num_elements] = total;
    }
}

size_t EstimatePrefixSumWorkspaceSizeBytes(size_t num_elements) {
    if (num_elements <= static_cast<size_t>(kElementsPerBlock)) {
        return 0;
    }
    const size_t num_blocks = (num_elements + kElementsPerBlock - 1) / kElementsPerBlock;
    return num_blocks * sizeof(int);
}

void PrefixSumDevice(const int* input, int* output, int* workspace, size_t num_elements) {
    if (num_elements == 0) {
        return;
    }
    const size_t num_blocks = (num_elements + kElementsPerBlock - 1) / kElementsPerBlock;

    if (num_blocks == 1) {
        SinglePassScanKernel<<<1, kBlockSize>>>(input, output, num_elements);
        return;
    }

    int* block_sums = workspace;

    BlockExclusiveScanKernel<<<num_blocks, kBlockSize>>>(input, output, block_sums, num_elements);
    BlockInclusiveScanKernel<<<1, kBlockSize>>>(block_sums, block_sums, num_blocks);
    AddOffsetsKernel<<<num_blocks, kBlockSize>>>(output, block_sums, num_elements);
    WriteLastElementKernel<<<1, 1>>>(output, block_sums, num_blocks, num_elements);
}
