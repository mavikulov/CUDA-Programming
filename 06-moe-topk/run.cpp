#include "moe_topk.cuh"

#include <cuda_helpers.h>

#include <cstddef>
#include <random>
#include <vector>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

namespace {

template <typename T>
struct Matrix {
    size_t stride;
    T* pointer;
};

template <typename T>
Matrix<T> AllocDeviceMatrix(size_t lines, size_t lineSize) {
    uint8_t* device_ptr = nullptr;
    size_t stride = 0;
    CheckStatus(cudaMallocPitch(reinterpret_cast<void**>(&device_ptr), &stride,
                                lineSize * sizeof(T), lines));
    return {.stride = stride, .pointer = reinterpret_cast<T*>(device_ptr)};
}

struct BenchmarkData {
    BenchmarkData(size_t num_tokens, size_t num_experts, size_t top_k)
        : NumTokens_(num_tokens), NumExperts_(num_experts), TopK_(top_k) {
        std::vector<__half> logits(num_tokens * num_experts);
        std::mt19937_64 rng(777);
        std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
        for (__half& x : logits) {
            x = __float2half(dist(rng));
        }

        LogitsDevice_ = AllocDeviceMatrix<__half>(num_tokens, num_experts);
        CheckStatus(cudaMemcpy2D(reinterpret_cast<void*>(LogitsDevice_.pointer),
                                 LogitsDevice_.stride, logits.data(), num_experts * sizeof(__half),
                                 num_experts * sizeof(__half), num_tokens, cudaMemcpyHostToDevice));

        IndicesDevice_ = AllocDeviceMatrix<int32_t>(num_tokens, top_k);
        WeightsDevice_ = AllocDeviceMatrix<__half>(num_tokens, top_k);
    }

    void DoBenchmark() {
        MoeTopK(NumTokens_, NumExperts_, TopK_, LogitsDevice_.pointer,
                LogitsDevice_.stride / sizeof(__half), IndicesDevice_.pointer,
                IndicesDevice_.stride / sizeof(int32_t), WeightsDevice_.pointer,
                WeightsDevice_.stride / sizeof(__half));
        CheckStatus(cudaGetLastError());
        CheckStatus(cudaDeviceSynchronize());
    }

    ~BenchmarkData() {
        CheckStatus(cudaFree(LogitsDevice_.pointer));
        CheckStatus(cudaFree(IndicesDevice_.pointer));
        CheckStatus(cudaFree(WeightsDevice_.pointer));
    }

private:
    size_t NumTokens_;
    size_t NumExperts_;
    size_t TopK_;
    Matrix<__half> LogitsDevice_;
    Matrix<int32_t> IndicesDevice_;
    Matrix<__half> WeightsDevice_;
};

}  // namespace

TEST_CASE("BenchmarkMoeTopK") {
    BenchmarkData large(8192, 128, 8);
    BenchmarkData small(64, 8, 2);

    BENCHMARK("MoeTopKLarge") {
        large.DoBenchmark();
    };
    BENCHMARK("MoeTopKSmall") {
        small.DoBenchmark();
    };
}
