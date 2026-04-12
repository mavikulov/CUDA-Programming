#include "puzzle.cuh"

#include <cuda_helpers.h>

#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace {

template <typename Value>
Value* AllocDeviceVector(size_t count) {
    Value* ptr_device = nullptr;

    CheckStatus(cudaMalloc(reinterpret_cast<void**>(&ptr_device), count * sizeof(Value)));

    return ptr_device;
}

std::vector<float> MapSpec(const std::vector<float>& data) {
    std::vector<float> result(data.size());

    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = data[index] + 10;
    }

    return result;
}

std::vector<float> ZipSpec(const std::vector<float>& left, const std::vector<float>& right) {
    std::vector<float> result(left.size());

    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = left[index] + right[index];
    }

    return result;
}

std::vector<float> GuardSpec(const std::vector<float>& data) {
    std::vector<float> result(data.size());

    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = data[index] + 10;
    }

    return result;
}

std::vector<float> BlockSpec(const std::vector<float>& data, float value) {
    std::vector<float> result(data.size());

    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = data[index] + value;
    }

    return result;
}

}  // namespace

TEST_CASE("Map") {
    const auto test_case = GENERATE(std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f});

    const std::vector<float> expected = MapSpec(test_case);

    auto* data_device = AllocDeviceVector<float>(test_case.size());
    auto* out_device = AllocDeviceVector<float>(test_case.size());
    CheckStatus(cudaMemcpy(data_device, test_case.data(), test_case.size() * sizeof(float),
                           cudaMemcpyHostToDevice));

    Map<<<1, test_case.size()>>>(data_device, out_device);

    std::vector<float> out_host(test_case.size());
    CheckStatus(cudaMemcpy(out_host.data(), out_device, test_case.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));

    REQUIRE(out_host == expected);

    CheckStatus(cudaFree(data_device));
    CheckStatus(cudaFree(out_device));
}

TEST_CASE("Zip") {
    struct TestCase {
        std::vector<float> left;
        std::vector<float> right;
    };

    // clang-format off
    const auto test_case = GENERATE(
        TestCase{{1.0f, 2.0f, 3.0f, 4.0f}, {1.0f, 2.0f, 3.0f, 4.0f}}
    );
    // clang-format on

    const std::vector<float> expected = ZipSpec(test_case.left, test_case.right);

    auto* left_device = AllocDeviceVector<float>(test_case.left.size());
    auto* right_device = AllocDeviceVector<float>(test_case.right.size());
    auto* out_device = AllocDeviceVector<float>(expected.size());
    CheckStatus(cudaMemcpy(left_device, test_case.left.data(),
                           test_case.left.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckStatus(cudaMemcpy(right_device, test_case.right.data(),
                           test_case.right.size() * sizeof(float), cudaMemcpyHostToDevice));

    Zip<<<1, expected.size()>>>(left_device, right_device, out_device);

    std::vector<float> out_host(expected.size());
    CheckStatus(cudaMemcpy(out_host.data(), out_device, out_host.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));

    REQUIRE(out_host == expected);

    CheckStatus(cudaFree(left_device));
    CheckStatus(cudaFree(right_device));
    CheckStatus(cudaFree(out_device));
}

TEST_CASE("Guard") {
    const auto test_case = GENERATE(std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f});

    const std::vector<float> expected = GuardSpec(test_case);

    auto* data_device = AllocDeviceVector<float>(test_case.size());
    auto* out_device = AllocDeviceVector<float>(test_case.size());
    CheckStatus(cudaMemcpy(data_device, test_case.data(), test_case.size() * sizeof(float),
                           cudaMemcpyHostToDevice));

    Guard<<<1, test_case.size() * 2>>>(data_device, out_device, test_case.size());

    std::vector<float> out_host(test_case.size());
    CheckStatus(cudaMemcpy(out_host.data(), out_device, test_case.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));

    REQUIRE(out_host == expected);

    CheckStatus(cudaFree(data_device));
    CheckStatus(cudaFree(out_device));
}

TEST_CASE("Block") {
    const std::vector<float> input{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const float value_to_add = 42.0f;

    const std::vector<float> expected = BlockSpec(input, value_to_add);

    auto* data_device = AllocDeviceVector<float>(input.size());
    auto* out_device = AllocDeviceVector<float>(input.size());
    CheckStatus(cudaMemcpy(data_device, input.data(), input.size() * sizeof(float),
                           cudaMemcpyHostToDevice));

    Block<<<4, 3>>>(data_device, out_device, value_to_add, input.size());

    std::vector<float> out_host(input.size());
    CheckStatus(cudaMemcpy(out_host.data(), out_device, input.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));

    REQUIRE(out_host == expected);

    CheckStatus(cudaFree(data_device));
    CheckStatus(cudaFree(out_device));
}
