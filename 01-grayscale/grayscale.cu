#include "grayscale.cuh"
#include <stdexcept>
#include <string>

void CheckStatus(const cudaError_t& status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status));
    }
}

Image AllocHostImage(size_t width, size_t height, size_t channels) {
    size_t stride = width * channels;
    uint8_t* pixels = (uint8_t*)malloc(width * height * channels * sizeof(uint8_t));
    return Image{pixels, width, height, stride, channels};
}

Image AllocDeviceImage(size_t width, size_t height, size_t channels) {
    Image img{};
    img.width = width;
    img.height = height;
    img.channels = channels;

    CheckStatus(
        cudaMallocPitch(&img.pixels, &img.stride, width * channels * sizeof(uint8_t), height));
    return img;
}

void CopyImageHostToDevice(const Image& src_host, Image& dst_device) {
    CheckStatus(cudaMemcpy2D(dst_device.pixels, dst_device.stride, src_host.pixels, src_host.stride,
                             src_host.width * src_host.channels * sizeof(uint8_t), src_host.height,
                             cudaMemcpyHostToDevice));
}

void CopyImageDeviceToHost(const Image& src_device, Image& dst_host) {
    CheckStatus(cudaMemcpy2D(dst_host.pixels, dst_host.stride, src_device.pixels, src_device.stride,
                             dst_host.width * dst_host.channels * sizeof(uint8_t), dst_host.height,
                             cudaMemcpyDeviceToHost));
}

__global__ void RGBToGrayKernel(const uint8_t* rgbImage, uint8_t* grayImage, size_t width,
                                size_t height, size_t rgbStride, size_t grayStride) {
    size_t xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (xIdx < width && yIdx < height) {
        const uint8_t* rgbRow = rgbImage + yIdx * rgbStride;
        uint8_t* grayRow = grayImage + yIdx * grayStride;
        size_t rgbIdx = xIdx * 3;
        uint8_t r = rgbRow[rgbIdx + 0];
        uint8_t g = rgbRow[rgbIdx + 1];
        uint8_t b = rgbRow[rgbIdx + 2];
        float grayValue = 0.299 * r + 0.587 * g + 0.114 * b;
        grayRow[xIdx] = static_cast<uint8_t>(grayValue);
    }
}

void ConvertToGrayscaleDevice(const Image& rgb_device_image, Image& gray_device_image) {
    dim3 block(16, 16);
    dim3 grid(((rgb_device_image.width + (block.x - 1)) / block.x),
              ((rgb_device_image.height + (block.y - 1)) / block.y));

    RGBToGrayKernel<<<grid, block>>>(rgb_device_image.pixels, gray_device_image.pixels,
                                     rgb_device_image.width, rgb_device_image.height,
                                     rgb_device_image.stride, gray_device_image.stride);
}

void FreeDeviceImage(const Image& image) {
    CheckStatus(cudaFree(image.pixels));
}

void FreeHostImage(const Image& image) {
    free(image.pixels);
}
