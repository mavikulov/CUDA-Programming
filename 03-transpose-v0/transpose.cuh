#include <cuda_fp16.h>

#include <cuda_helpers.h>

__global__ void TransposeKernel(const __half* input_device, size_t input_stride,
                                __half* output_device, size_t output_stride, size_t num_rows,
                                size_t num_cols) {
    __shared__ __half temp[32][32 + 1];  // blockDim = 32
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((col < num_cols) && (row < num_rows)) {
        temp[threadIdx.y][threadIdx.x] = input_device[row * input_stride + col];
    }

    __syncthreads();

    int out_col = blockIdx.y * 32 + threadIdx.x;
    int out_row = blockIdx.x * 32 + threadIdx.y;
    if (out_col < num_rows && out_row < num_cols) {
        output_device[out_row * output_stride + out_col] = temp[threadIdx.x][threadIdx.y];
    }
}

void TransposeDevice(const __half* input_device, size_t input_stride, __half* output_device,
                     size_t output_stride, size_t num_rows, size_t num_cols) {
    auto blockSize = 32;
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((num_cols + blockSize - 1) / blockSize, (num_rows + blockSize - 1) / blockSize);
    TransposeKernel<<<gridDim, blockDim>>>(input_device, input_stride, output_device, output_stride,
                                           num_rows, num_cols);
}
