#include "quaternions.cuh"
#include "quaternions_helpers.h"

#include <cstddef>
#include <random>
#include <vector>

#include <cuda_helpers.h>
#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace {
struct BenchmarkData {
    BenchmarkData(size_t line_size, size_t lines_count, float std_dev)
        : line_size_(line_size), lines_count_(lines_count) {
        std::seed_seq seed{31337};
        std::mt19937 mt{seed};

        std::normal_distribution<> normal_dist{};

        auto gen_normal_float = [&]() -> float { return normal_dist(mt) * std_dev; };

        std::vector<Quaternion> input_matrix(lines_count * line_size);
        for (size_t line_idx = 0; line_idx < lines_count; ++line_idx) {
            for (size_t in_line_idx = 0; in_line_idx < line_size; ++in_line_idx) {
                input_matrix[line_idx * line_size + in_line_idx] =
                    quaternions::GenQuaternion(gen_normal_float);
            }
        }

        CheckStatus(cudaStreamCreate(&stream_));
        CheckStatus(cudaGetLastError());

        input_matrix_device_ = quaternions::AllocDeviceMatrix<Quaternion>(lines_count, line_size);
        CheckStatus(cudaMemcpy2D(reinterpret_cast<void*>(input_matrix_device_.pointer),
                                 input_matrix_device_.stride * sizeof(Quaternion),
                                 input_matrix.data(), line_size * sizeof(Quaternion),
                                 line_size * sizeof(Quaternion), lines_count,
                                 cudaMemcpyHostToDevice));

        out_vector_device_ = quaternions::AllocDeviceVector<Quaternion>(lines_count);
    }

    void DoBenchmark() {
        QuaternionsReduce(lines_count_, line_size_, input_matrix_device_.pointer,
                          input_matrix_device_.stride, out_vector_device_, stream_);
        CheckStatus(cudaGetLastError());
        CheckStatus(cudaDeviceSynchronize());
    }

    ~BenchmarkData() {
        CheckStatus(cudaStreamDestroy(stream_));
        CheckStatus(cudaFree(out_vector_device_));
        CheckStatus(cudaFree(input_matrix_device_.pointer));
    }

private:
    size_t line_size_;
    size_t lines_count_;
    quaternions::Matrix<Quaternion> input_matrix_device_;
    Quaternion* out_vector_device_;
    cudaStream_t stream_;
};
}  // namespace

TEST_CASE("BenchmarkQuaternions") {
    BenchmarkData benchV0(4096, 1, 10);
    BenchmarkData benchV1(2048, 100, 20);
    BenchmarkData benchV2(1024, 8192, 40);

    BENCHMARK("QuaternionsV0") {
        benchV0.DoBenchmark();
    };

    BENCHMARK("QuaternionsV1") {
        benchV1.DoBenchmark();
    };

    BENCHMARK("QuaternionsV2") {
        benchV2.DoBenchmark();
    };
}
