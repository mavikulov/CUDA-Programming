#include "prefix_sum.cuh"

#include <cstddef>
#include <vector>

#include <cuda_helpers.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace {

struct BenchmarkData {
    int* input_device = nullptr;
    int* output_device = nullptr;
    int* workspace_device = nullptr;
};

void DoBenchmark(size_t num_elements, const char* bench_name) {
    std::vector<BenchmarkData> datas;
    size_t data_generated_bytes = 0;
    while (data_generated_bytes < GetL2CacheSizeBytes() || datas.size() < 2) {
        BenchmarkData data{};

        CheckStatus(cudaMalloc(&data.input_device, num_elements * sizeof(int)));
        const std::vector<int> input_host(num_elements, 1);
        CheckStatus(cudaMemcpy(data.input_device, input_host.data(), num_elements * sizeof(int),
                               cudaMemcpyHostToDevice));
        CheckStatus(cudaMalloc(&data.output_device, (num_elements + 1) * sizeof(int)));
        CheckStatus(
            cudaMalloc(&data.workspace_device, EstimatePrefixSumWorkspaceSizeBytes(num_elements)));

        datas.push_back(data);
        data_generated_bytes += num_elements * sizeof(int);
    }

    size_t data_idx = 0;
    BENCHMARK(bench_name) {
        for (size_t iter = 0; iter < 10; ++iter) {
            BenchmarkData& data = datas[data_idx];
            PrefixSumDevice(data.input_device, data.output_device, data.workspace_device,
                            num_elements);
            data_idx = (data_idx + 1) % datas.size();
        }
        CheckStatus(cudaDeviceSynchronize());
    };

    for (auto& data : datas) {
        CheckStatus(cudaFree(data.input_device));
        CheckStatus(cudaFree(data.output_device));
        CheckStatus(cudaFree(data.workspace_device));
    }
}

}  // namespace

TEST_CASE("Benchmark") {
    DoBenchmark(1000, "PrefixSumSmall");
    DoBenchmark(1'000'000, "PrefixSumLarge");
}
