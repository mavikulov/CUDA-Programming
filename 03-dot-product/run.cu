#include "dot_product.cuh"

#include <cstddef>
#include <vector>

#include <cuda_helpers.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace {

struct BenchmarkData {
    float* lhs_device = nullptr;
    float* rhs_device = nullptr;
    float* workspace_device = nullptr;
    float* out_device = nullptr;
};

}  // namespace

TEST_CASE("Benchmark") {
    const size_t kNumElements = 1'000'000;

    std::vector<BenchmarkData> datas;
    size_t data_generated_bytes = 0;
    while (data_generated_bytes < GetL2CacheSizeBytes() || datas.size() < 2) {
        BenchmarkData data{};
        CheckStatus(
            cudaMalloc(reinterpret_cast<void**>(&data.lhs_device), kNumElements * sizeof(float)));
        CheckStatus(
            cudaMalloc(reinterpret_cast<void**>(&data.rhs_device), kNumElements * sizeof(float)));
        CheckStatus(cudaMalloc(reinterpret_cast<void**>(&data.workspace_device),
                               EstimateDotProductWorkspaceSizeBytes(kNumElements)));
        CheckStatus(cudaMalloc(reinterpret_cast<void**>(&data.out_device), sizeof(float)));

        const std::vector<float> lhs_host(kNumElements, 1.0f);
        const std::vector<float> rhs_host(kNumElements, 1.0f);
        CheckStatus(cudaMemcpy(data.lhs_device, lhs_host.data(), lhs_host.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
        CheckStatus(cudaMemcpy(data.rhs_device, rhs_host.data(), rhs_host.size() * sizeof(float),
                               cudaMemcpyHostToDevice));

        datas.push_back(data);
        data_generated_bytes += kNumElements * 2 * sizeof(float) +
                                EstimateDotProductWorkspaceSizeBytes(kNumElements) + sizeof(float);
    }

    size_t data_idx = 0;
    BENCHMARK("DotProduct") {
        for (size_t iter = 0; iter < 50; ++iter) {
            BenchmarkData& data = datas[data_idx];
            DotProduct(data.lhs_device, data.rhs_device, kNumElements, data.workspace_device,
                       data.out_device);
            data_idx = (data_idx + 1) % datas.size();
        }
        CheckStatus(cudaDeviceSynchronize());
    };

    for (auto& data : datas) {
        CheckStatus(cudaFree(data.lhs_device));
        CheckStatus(cudaFree(data.rhs_device));
        CheckStatus(cudaFree(data.workspace_device));
        CheckStatus(cudaFree(data.out_device));
    }
}
