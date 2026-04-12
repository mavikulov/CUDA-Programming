#include "transpose.cuh"

#include <cuda_helpers.h>

#include <iostream>
#include <random>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

std::ostream& operator<<(std::ostream& os, const __half& h) {
    os << __half2float(h);
    return os;
}

namespace {

struct TestCase {
    std::vector<__half> input;
    std::vector<__half> expected;
    size_t rows;
    size_t cols;
};

void DoTransposeTest(const TestCase& test_case) {
    __half* input_device = nullptr;
    size_t input_pitch = 0;
    __half* output_device = nullptr;
    size_t output_pitch = 0;
    CheckStatus(cudaMallocPitch(&input_device, &input_pitch, test_case.cols * sizeof(__half),
                                test_case.rows));
    CheckStatus(cudaMallocPitch(&output_device, &output_pitch, test_case.rows * sizeof(__half),
                                test_case.cols));
    CheckStatus(cudaMemcpy2D(input_device, input_pitch, test_case.input.data(),
                             test_case.cols * sizeof(__half), test_case.cols * sizeof(__half),
                             test_case.rows, cudaMemcpyHostToDevice));

    TransposeDevice(input_device, input_pitch / sizeof(__half), output_device,
                    output_pitch / sizeof(__half), test_case.rows, test_case.cols);

    std::vector<__half> output_host(test_case.rows * test_case.cols);
    CheckStatus(cudaMemcpy2D(output_host.data(), test_case.rows * sizeof(__half), output_device,
                             output_pitch, test_case.rows * sizeof(__half), test_case.cols,
                             cudaMemcpyDeviceToHost));

    REQUIRE(output_host == test_case.expected);

    CheckStatus(cudaFree(input_device));
    CheckStatus(cudaFree(output_device));
}

TestCase GenerateRandomTestCase(size_t rows, size_t cols) {
    static std::mt19937_64 gen{42};

    const size_t num_elements = rows * cols;
    std::vector<__half> values(num_elements);
    std::vector<__half> transposed(num_elements);

    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (size_t index = 0; index < num_elements; ++index) {
        values[index] = static_cast<__half>(distribution(gen));
        const size_t col = index % cols;
        const size_t row = index / cols;
        transposed[col * rows + row] = values[index];
    }

    return TestCase{
        .input = std::move(values), .expected = std::move(transposed), .rows = rows, .cols = cols};
}

}  // namespace

TEST_CASE("Transpose") {
    SECTION("Basic") {
        const auto test_case = GENERATE(
            TestCase{.input = {1.0f}, .expected = {1.0f}, .rows = 1, .cols = 1},
            TestCase{
                .input = {1.0f, 2.0f, 3.0f}, .expected = {1.0f, 2.0f, 3.0f}, .rows = 3, .cols = 1},
            TestCase{
                .input = {1.0f, 2.0f, 3.0f}, .expected = {1.0f, 2.0f, 3.0f}, .rows = 1, .cols = 3},
            TestCase{.input = {1.0f, 2.0f, 3.0f, 4.0f},
                     .expected = {1.0f, 3.0f, 2.0f, 4.0f},
                     .rows = 2,
                     .cols = 2});

        DoTransposeTest(test_case);
    }

    SECTION("Large") {
        const auto test_case =
            GENERATE(GenerateRandomTestCase(100, 200), GenerateRandomTestCase(1000, 1000));

        DoTransposeTest(test_case);
    }
}
