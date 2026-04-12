#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_helpers.h>

struct Image {
    uint8_t* pixels;
    size_t width;
    size_t height;
    size_t stride;
    size_t channels;
};

Image AllocHostImage(size_t width, size_t height, size_t channels);
Image AllocDeviceImage(size_t width, size_t height, size_t channels);
void CopyImageHostToDevice(const Image& src_host, Image& dst_device);
void CopyImageDeviceToHost(const Image& src_device, Image& dst_host);
void ConvertToGrayscaleDevice(const Image& rgb_device_image, Image& gray_device_image);
void FreeDeviceImage(const Image& image);
void FreeHostImage(const Image& image);
