#include "softmax.cuh"

#include <cstddef>
#include <cstddef>
#include <random>
#include <vector>

#include <cuda_helpers.h>
#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

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

struct BenchmarkData {
    BenchmarkData(size_t line_size, size_t lines_count, float std_dev)
        : line_size_(line_size), lines_count_(lines_count) {
        std::seed_seq seed{31337};
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

        CheckStatus(cudaStreamCreate(&stream_));
        CheckStatus(cudaGetLastError());

        input_matrix_device_ = AllocDeviceMatrix<float>(lines_count, line_size);

        CheckStatus(cudaMemcpy2D(input_matrix_device_.pointer,
                                 input_matrix_device_.stride * sizeof(float), input_matrix.data(),
                                 line_size * sizeof(float), line_size * sizeof(float), lines_count,
                                 cudaMemcpyHostToDevice));
        out_matrix_device_ = AllocDeviceMatrix<float>(lines_count, line_size);
    }

    void DoBenchmark() {
        Softmax(lines_count_, line_size_, input_matrix_device_.pointer, input_matrix_device_.stride,
                out_matrix_device_.pointer, out_matrix_device_.stride, stream_);
        CheckStatus(cudaGetLastError());
        CheckStatus(cudaDeviceSynchronize());
    }

    ~BenchmarkData() {
        CheckStatus(cudaStreamDestroy(stream_));
        CheckStatus(cudaFree(out_matrix_device_.pointer));
        CheckStatus(cudaFree(input_matrix_device_.pointer));
    }

private:
    size_t line_size_;
    size_t lines_count_;
    Matrix<float> input_matrix_device_;
    Matrix<float> out_matrix_device_;
    cudaStream_t stream_;
};
}  // namespace

TEST_CASE("BenchmarkSoftmaxLong") {
    BenchmarkData largeV0(128000, 1, 1);
    BenchmarkData largeV1(128000, 49, 40);
    BenchmarkData largeV2(85172, 843, 40);

    BENCHMARK("SoftmaxLongV0") {
        largeV0.DoBenchmark();
    };

    BENCHMARK("SoftmaxLongV1") {
        largeV1.DoBenchmark();
    };

    BENCHMARK("SoftmaxLongV2") {
        largeV2.DoBenchmark();
    };
}
