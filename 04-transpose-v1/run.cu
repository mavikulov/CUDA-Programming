#include "transpose.cuh"

#include <cstddef>
#include <vector>

#include <cuda_helpers.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace {

struct BenchmarkData {
    __half* input_device = nullptr;
    size_t input_stride = 0;
    __half* output_device = nullptr;
    size_t output_stride = 0;
};

}  // namespace

TEST_CASE("Benchmark") {
    const size_t kRows = 1024;
    const size_t kCols = 1024;

    std::vector<BenchmarkData> datas;
    size_t data_generated_bytes = 0;
    while (data_generated_bytes < GetL2CacheSizeBytes() || datas.size() < 2) {
        BenchmarkData data{};

        size_t input_pitch = 0;
        CheckStatus(
            cudaMallocPitch(&data.input_device, &input_pitch, kRows * sizeof(__half), kCols));
        data.input_stride = input_pitch / sizeof(__half);

        size_t output_pitch = 0;
        CheckStatus(
            cudaMallocPitch(&data.output_device, &output_pitch, kCols * sizeof(__half), kRows));
        data.output_stride = output_pitch / sizeof(__half);

        const std::vector<__half> input_host(kRows * kCols, 1.0f);
        CheckStatus(cudaMemcpy2D(data.input_device, input_pitch, input_host.data(),
                                 kRows * sizeof(__half), kRows * sizeof(__half), kCols,
                                 cudaMemcpyHostToDevice));

        datas.push_back(data);
        data_generated_bytes += kRows * kCols * sizeof(__half);
    }

    size_t data_idx = 0;
    BENCHMARK("Transpose") {
        for (size_t iter = 0; iter < 100; ++iter) {
            BenchmarkData& data = datas[data_idx];
            TransposeDevice(data.input_device, data.input_stride, data.output_device,
                            data.output_stride, kRows, kCols);
            data_idx = (data_idx + 1) % datas.size();
        }
        CheckStatus(cudaDeviceSynchronize());
    };

    for (auto& data : datas) {
        CheckStatus(cudaFree(data.input_device));
        CheckStatus(cudaFree(data.output_device));
    }
}
