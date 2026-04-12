#include "gemm.cuh"
#include "test_utils.h"

#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("GEMM") {
    const auto test_case = GENERATE(
        TestCase{.a = {{{2.0f}}, MatrixLayout::RowMajor},
                 .b = {{{3.0f}}, MatrixLayout::ColMajor},
                 .c = {{{1.0f}}, MatrixLayout::ColMajor},
                 .expected = {{{7.0f}}, MatrixLayout::ColMajor},
                 .inplace = false,
                 .alpha = 1.0f,
                 .beta = 1.0f},
        TestCase{.a = {{{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}}, MatrixLayout::RowMajor},
                 .b = {{{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}}, MatrixLayout::ColMajor},
                 .c = {{{1.0f}}, MatrixLayout::ColMajor},
                 .expected = {{{29.5f}}, MatrixLayout::ColMajor},
                 .inplace = true,
                 .alpha = 0.5f,
                 .beta = 2.0f},
        TestCase{.a = {{{1.0f, 2.0f}, {-1.0f, -2.0f}}, MatrixLayout::RowMajor},
                 .b = {{{-3.0f, 4.0f}, {5.0f, 6.0f}}, MatrixLayout::ColMajor},
                 .c = {{{0.0f, 0.0f}, {0.0f, 0.0f}}, MatrixLayout::ColMajor},
                 .expected = {{{5.0f, -5.0f}, {17.0f, -17.0f}}, MatrixLayout::ColMajor},
                 .inplace = false,
                 .alpha = 1.0f,
                 .beta = 0.0f},
        GenerateRandomTestCase(30, 30, 30, MatrixLayout::RowMajor, MatrixLayout::ColMajor,
                               MatrixLayout::ColMajor, false, 2.0f, 1.0f),
        GenerateRandomTestCase(200, 80, 200, MatrixLayout::RowMajor, MatrixLayout::ColMajor,
                               MatrixLayout::ColMajor, false, -2.0f, 0.5f));

    DeviceMatrix a_device = test_case.a.ToGPU();
    DeviceMatrix b_device = test_case.b.ToGPU();
    DeviceMatrix c_device = test_case.c.ToGPU();
    DeviceMatrix d_device = test_case.inplace ? c_device : AllocAlike(c_device);

    REQUIRE(a_device.cols == b_device.rows);
    REQUIRE((c_device.rows == a_device.rows && c_device.cols == b_device.cols));
    REQUIRE((c_device.rows == d_device.rows && c_device.cols == d_device.cols));
    REQUIRE(c_device.layout == d_device.layout);

    GEMM(a_device, b_device, c_device, d_device, test_case.alpha, test_case.beta);
    CheckStatus(cudaGetLastError());

    HostMatrix out = HostMatrix::FromGPU(d_device);
    for (size_t row = 0; row < out.rows; ++row) {
        for (size_t col = 0; col < out.cols; ++col) {
            const size_t index =
                out.layout == MatrixLayout::RowMajor ? row * out.cols + col : col * out.rows + row;
            const float actual = out.data[index];
            const float expected = test_case.expected.data[index];
            INFO("row = " << row << " col = " << col);
            CHECK_THAT(actual, Catch::Matchers::WithinRel(expected, 1e-4f));
        }
    }

    CheckStatus(cudaFree(a_device.data));
    CheckStatus(cudaFree(b_device.data));
    CheckStatus(cudaFree(c_device.data));
    if (!test_case.inplace) {
        CheckStatus(cudaFree(d_device.data));
    }
}
