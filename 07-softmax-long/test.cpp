#include "softmax.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_helpers.h>
#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace {
template <typename T>
struct Matrix {
    size_t stride;
    T* pointer;
};

template <typename T>
Matrix<T> AllocDeviceMatrix(size_t lines, size_t line_size) {
    uint8_t* device_ptr = nullptr;
    size_t stride = 0;
    CheckStatus(cudaMallocPitch(reinterpret_cast<void**>(&device_ptr), &stride,
                                line_size * sizeof(T), lines));
    return {.stride = stride / sizeof(T), .pointer = reinterpret_cast<T*>(device_ptr)};
}

void RefSoftmax(size_t rows, size_t cols, const float* inp, float* out) {
    for (size_t row_idx = 0; row_idx < rows; ++row_idx, inp += cols, out += cols) {
        // calculate via maximum for numeric stability
        float maximum = std::numeric_limits<float>::lowest();
        for (size_t col_idx = 0; col_idx < cols; ++col_idx) {
            maximum = std::max(maximum, inp[col_idx]);
        }

        float norm = 0.f;
        for (size_t col_idx = 0; col_idx < cols; ++col_idx) {
            norm += std::exp(inp[col_idx] - maximum);
        }

        for (size_t col_idx = 0; col_idx < cols; ++col_idx) {
            out[col_idx] = std::exp(inp[col_idx] - maximum) / norm;
        }
    }
}

void DoSoftmaxTest(size_t line_size, size_t lines_count, float std_dev, size_t seed_value) {
    std::seed_seq seed{seed_value};
    std::mt19937 mt{seed};

    std::normal_distribution<> normal_dist{};
    std::uniform_real_distribution<> uniform_dist{-1000.f, 1000.f};

    std::vector<float> input_matrix(line_size * lines_count);
    for (size_t line_idx = 0; line_idx < lines_count; ++line_idx) {
        const float mean = uniform_dist(mt);
        for (size_t in_line_idx = 0; in_line_idx < line_size; ++in_line_idx) {
            input_matrix[line_idx * line_size + in_line_idx] = normal_dist(mt) * std_dev + mean;
        }
    }

    std::vector<float> out_matrix_ref(line_size * lines_count);
    RefSoftmax(lines_count, line_size, input_matrix.data(), out_matrix_ref.data());

    cudaStream_t stream;
    CheckStatus(cudaStreamCreate(&stream));

    auto input_matrix_device = AllocDeviceMatrix<float>(lines_count, line_size);
    CheckStatus(cudaMemcpy2D(reinterpret_cast<void*>(input_matrix_device.pointer),
                             input_matrix_device.stride * sizeof(float), input_matrix.data(),
                             line_size * sizeof(float), line_size * sizeof(float), lines_count,
                             cudaMemcpyHostToDevice));

    auto out_matrix_device = AllocDeviceMatrix<float>(lines_count, line_size);

    INFO("ROWS = " << lines_count << " COLS = " << line_size << " STD_DEV = " << std_dev
                   << " SEED = " << seed_value);

    Softmax(lines_count, line_size, input_matrix_device.pointer, input_matrix_device.stride,
            out_matrix_device.pointer, out_matrix_device.stride, stream);

    CheckStatus(cudaGetLastError());
    CheckStatus(cudaStreamSynchronize(stream));

    std::vector<float> out_matrix(line_size * lines_count);
    CheckStatus(cudaMemcpy2D(out_matrix.data(), line_size * sizeof(float),
                             reinterpret_cast<void*>(out_matrix_device.pointer),
                             out_matrix_device.stride * sizeof(float), line_size * sizeof(float),
                             lines_count, cudaMemcpyDeviceToHost));

    REQUIRE_THAT(out_matrix, Catch::Matchers::Approx(out_matrix_ref).margin(1e-4));

    CheckStatus(cudaFree(input_matrix_device.pointer));
    CheckStatus(cudaFree(out_matrix_device.pointer));
    CheckStatus(cudaStreamDestroy(stream));
}
}  // namespace

TEST_CASE("Softmax") {
    SECTION("Large") {
        const auto line_size = GENERATE(4096, 50284, 128000);
        const auto lines_count = GENERATE(1, 4, 47, 100);
        const auto std_dev = GENERATE(5, 40, 100);

        DoSoftmaxTest(line_size, lines_count, std_dev, 42);
    }
}
