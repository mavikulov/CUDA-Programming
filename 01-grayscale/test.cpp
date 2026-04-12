#include "grayscale.cuh"

#include <cuda_helpers.h>

#include <filesystem>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#pragma GCC diagnostic pop

namespace {

Image LoadSpecHostImage(const std::filesystem::path& path) {
    int width = 0;
    int height = 0;
    const int channels = 3;

    uint8_t* raw_pixels = stbi_load(path.c_str(), &width, &height, nullptr, channels);
    REQUIRE(raw_pixels != nullptr);

    Image image{.pixels = raw_pixels,
                .width = static_cast<size_t>(width),
                .height = static_cast<size_t>(height),
                .stride = static_cast<size_t>(width * channels),
                .channels = static_cast<size_t>(channels)};

    return image;
}

void FreeSpecHostImage(const Image& image) {
    stbi_image_free(image.pixels);
}

void ConvertToGrayscaleHostSpec(const Image& rgb_image, Image& gray_image) {
    REQUIRE(rgb_image.width == gray_image.width);
    REQUIRE(rgb_image.height == gray_image.height);
    REQUIRE(rgb_image.channels == 3);
    REQUIRE(gray_image.channels == 1);

    for (size_t y = 0; y < rgb_image.height; ++y) {
        for (size_t x = 0; x < rgb_image.width; ++x) {
            const uint8_t* in_ptr =
                rgb_image.pixels + y * rgb_image.stride + rgb_image.channels * x;
            uint8_t* out_ptr = gray_image.pixels + y * gray_image.stride + x;

            uint8_t r = in_ptr[0];
            uint8_t g = in_ptr[1];
            uint8_t b = in_ptr[2];
            *out_ptr = static_cast<uint8_t>(r * 0.299f + g * 0.587f + b * 0.114f);
        }
    }
}

void EnsurePtrIsDevice(const uint8_t* ptr) {
    cudaPointerAttributes attr{};
    CheckStatus(cudaPointerGetAttributes(&attr, ptr));

    REQUIRE(attr.type == cudaMemoryTypeDevice);
}

}  // namespace

TEST_CASE("AllocImage") {
    size_t width = GENERATE(64, 128);
    size_t height = GENERATE(64, 128);
    size_t channels = GENERATE(1, 3);

    SECTION("AllocDeviceImage") {
        Image image = AllocDeviceImage(width, height, channels);

        REQUIRE(image.width == width);
        REQUIRE(image.height == height);
        REQUIRE(image.channels == channels);
        EnsurePtrIsDevice(image.pixels);

        FreeDeviceImage(image);
    }

    SECTION("AllocHostImage") {
        Image image = AllocHostImage(width, height, channels);

        REQUIRE(image.width == width);
        REQUIRE(image.height == height);
        REQUIRE(image.channels == channels);

        FreeHostImage(image);
    }
}

using Catch::Matchers::WithinAbs;

TEST_CASE("ConvertToGrayscale") {
    Image src_host_image = LoadSpecHostImage(std::filesystem::path(__FILE__).parent_path() /
                                             "test_data" / "huang.png");
    const size_t width = src_host_image.width;
    const size_t height = src_host_image.height;
    Image dst_host_image_expected = AllocHostImage(width, height, 1);
    ConvertToGrayscaleHostSpec(src_host_image, dst_host_image_expected);

    Image src_device_image = AllocDeviceImage(width, height, src_host_image.channels);
    Image dst_device_image = AllocDeviceImage(width, height, 1);
    Image dst_host_image = AllocHostImage(width, height, 1);

    CopyImageHostToDevice(src_host_image, src_device_image);
    ConvertToGrayscaleDevice(src_device_image, dst_device_image);
    CheckStatus(cudaGetLastError());
    CopyImageDeviceToHost(dst_device_image, dst_host_image);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            const int actual = dst_host_image.pixels[y * dst_host_image.stride + x];
            const int expected =
                dst_host_image_expected.pixels[y * dst_host_image_expected.stride + x];

            INFO("x = " << x << " y = " << y);
            CHECK_THAT(actual, WithinAbs(expected, 1.0));
        }
    }

    CheckStatus(cudaDeviceSynchronize());

    FreeHostImage(dst_host_image_expected);
    FreeHostImage(dst_host_image);
    FreeDeviceImage(dst_device_image);
    FreeDeviceImage(src_device_image);
    FreeSpecHostImage(src_host_image);
}
