#include "prefix_sum.cuh"

#include <cuda_helpers.h>

#include <random>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace {

struct TestCase {
    std::vector<int> input;
    std::vector<int> expected;
};

void DoPrefixSumTest(const TestCase& test_case) {
    REQUIRE(test_case.input.size() + 1 == test_case.expected.size());
    int* input_device = nullptr;
    int* output_device = nullptr;
    int* workspace_device = nullptr;
    CheckStatus(cudaMalloc(&input_device, test_case.input.size() * sizeof(int)));
    CheckStatus(cudaMemcpy(input_device, test_case.input.data(),
                           test_case.input.size() * sizeof(int), cudaMemcpyHostToDevice));
    CheckStatus(cudaMalloc(&output_device, test_case.expected.size() * sizeof(int)));
    CheckStatus(
        cudaMalloc(&workspace_device, EstimatePrefixSumWorkspaceSizeBytes(test_case.input.size())));

    PrefixSumDevice(input_device, output_device, workspace_device, test_case.input.size());

    std::vector<int> output_host(test_case.expected.size());
    CheckStatus(cudaMemcpy(output_host.data(), output_device,
                           test_case.expected.size() * sizeof(int), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < output_host.size(); ++i) {
        INFO("i = " << i);
        CHECK(output_host[i] == test_case.expected[i]);
    }

    CheckStatus(cudaFree(input_device));
    CheckStatus(cudaFree(output_device));
    CheckStatus(cudaFree(workspace_device));
}

TestCase GenerateRandomTestCase(size_t num_elements) {
    static std::mt19937_64 gen{42};

    std::vector<int> input(num_elements);
    std::vector<int> output(num_elements + 1);

    std::uniform_int_distribution<int> distribution(-10, 10);
    for (size_t index = 0; index < num_elements; ++index) {
        input[index] = distribution(gen);
        output[index + 1] = output[index] + input[index];
    }

    return TestCase{.input = std::move(input), .expected = std::move(output)};
}

}  // namespace

TEST_CASE("PrefixSum") {
    SECTION("Basic") {
        const auto test_case = GENERATE(TestCase{.input = {1}, .expected = {0, 1}},
                                        TestCase{.input = {1, 2, 3}, .expected = {0, 1, 3, 6}},
                                        TestCase{
                                            .input = {1, 2, 3, 4},
                                            .expected = {0, 1, 3, 6, 10},
                                        },
                                        GenerateRandomTestCase(33));

        DoPrefixSumTest(test_case);
    }

    SECTION("Large") {
        const auto test_case =
            GENERATE(GenerateRandomTestCase(1025), GenerateRandomTestCase(1'000'000),
                     GenerateRandomTestCase(2'097'169));

        DoPrefixSumTest(test_case);
    }
}

TEST_CASE("EstimatePrefixSumWorkspaceSizeBytes") {
    SECTION("NoAllocationsOnSmallArrays") {
        REQUIRE(EstimatePrefixSumWorkspaceSizeBytes(1) == 0);
        REQUIRE(EstimatePrefixSumWorkspaceSizeBytes(1024) == 0);
    }

    SECTION("Large") {
        REQUIRE(EstimatePrefixSumWorkspaceSizeBytes(1'000'000) <= 4096);
    }
}
