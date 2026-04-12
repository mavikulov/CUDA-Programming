#include "gemm.cuh"
#include "test_utils.h"

#include <cstddef>
#include <vector>

#include <cuda_helpers.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace {
struct BenchmarkData {
    DeviceMatrix a;
    DeviceMatrix b;
    DeviceMatrix c;
    DeviceMatrix d;
    float alpha = 0.0f;
    float beta = 0.0f;
};
}  // namespace

TEST_CASE("Benchmark") {
    const size_t kRows = 1024;
    const size_t kCols = 1024;

    std::vector<BenchmarkData> datas;
    size_t data_generated_bytes = 0;
    while (data_generated_bytes < GetL2CacheSizeBytes() || datas.size() < 2) {
        std::vector<std::vector<__half>> host_data(kRows, std::vector<__half>(kCols, 1.0f));
        HostMatrix a_host{host_data, MatrixLayout::RowMajor};
        HostMatrix b_host{host_data, MatrixLayout::ColMajor};
        HostMatrix c_host{host_data, MatrixLayout::ColMajor};
        DeviceMatrix a_device = a_host.ToGPU();
        DeviceMatrix b_device = b_host.ToGPU();
        DeviceMatrix c_device = c_host.ToGPU();
        DeviceMatrix d_device = AllocAlike(c_device);

        BenchmarkData data{
            .a = a_device,
            .b = b_device,
            .c = c_device,
            .d = d_device,
            .alpha = 1.0f,
            .beta = 0.5f,
        };
        datas.push_back(data);
        data_generated_bytes += kRows * kCols * sizeof(__half) * 3;
    }

    size_t data_idx = 0;
    BENCHMARK("GEMM") {
        for (size_t iter = 0; iter < 50; ++iter) {
            BenchmarkData& data = datas[data_idx];
            GEMM(data.a, data.b, data.c, data.d, data.alpha, data.beta);
            data_idx = (data_idx + 1) % datas.size();
        }
        CheckStatus(cudaDeviceSynchronize());
    };

    for (auto& data : datas) {
        CheckStatus(cudaFree(data.a.data));
        CheckStatus(cudaFree(data.b.data));
        CheckStatus(cudaFree(data.c.data));
        CheckStatus(cudaFree(data.d.data));
    }
}
