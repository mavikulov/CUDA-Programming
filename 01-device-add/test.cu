#include "add.cuh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace {

struct TestCase {
    std::vector<float> input_left;
    std::vector<float> input_right;
    std::vector<float> expected;
};

void EnsurePtrIsDevice(const float* ptr) {
    cudaPointerAttributes attr{};
    CheckStatus(cudaPointerGetAttributes(&attr, ptr));

    REQUIRE(attr.type == cudaMemoryTypeDevice);
}

}  // namespace

TEST_CASE("Add") {
    const auto test_case =
        GENERATE(TestCase{{2.0f}, {3.0f}, {5.0f}},
                 TestCase{{1.0f, 2.0f, 3.0f}, {3.0f, 2.0f, 1.0f}, {4.0f, 4.0f, 4.0f}},
                 TestCase{std::vector<float>(1'000'000, 1.0f), std::vector<float>(1'000'000, -0.5f),
                          std::vector<float>(1'000'000, 0.5f)});

    const size_t num_elements = test_case.input_left.size();

    float* left_device = AllocDeviceVector(num_elements);
    EnsurePtrIsDevice(left_device);
    float* right_device = AllocDeviceVector(num_elements);
    EnsurePtrIsDevice(right_device);
    float* out_device = AllocDeviceVector(num_elements);
    EnsurePtrIsDevice(out_device);

    CopyHostVectorToDevice(test_case.input_left, left_device);
    CopyHostVectorToDevice(test_case.input_right, right_device);

    AddDeviceVectors(left_device, right_device, out_device, num_elements);

    std::vector<float> out = CopyDeviceVectorToHost(out_device, num_elements);

    FreeDeviceVector(out_device);
    FreeDeviceVector(right_device);
    FreeDeviceVector(left_device);

    REQUIRE(out == test_case.expected);
}
