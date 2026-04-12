#include "dot_product.cuh"

#include <cuda_helpers.h>

#include <vector>
#include <random>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace {

float DotProductSpec(const std::vector<float>& lhs_host, const std::vector<float>& rhs_host) {
    float accumulator = 0.0f;

    REQUIRE(lhs_host.size() == rhs_host.size());

    for (size_t index = 0; index < lhs_host.size(); ++index) {
        accumulator += lhs_host[index] * rhs_host[index];
    }

    return accumulator;
}

void DoDotProductTest(const std::vector<float>& lhs_host, const std::vector<float>& rhs_host) {
    REQUIRE(lhs_host.size() == rhs_host.size());

    float* lhs_device = nullptr;
    float* rhs_device = nullptr;
    float* workspace_device = nullptr;
    float* out_device = nullptr;
    CheckStatus(cudaMalloc(&lhs_device, lhs_host.size() * sizeof(float)));
    CheckStatus(cudaMalloc(&rhs_device, rhs_host.size() * sizeof(float)));
    CheckStatus(
        cudaMalloc(&workspace_device, EstimateDotProductWorkspaceSizeBytes(lhs_host.size())));
    CheckStatus(cudaMalloc(&out_device, sizeof(float)));
    CheckStatus(cudaMemcpy(lhs_device, lhs_host.data(), lhs_host.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
    CheckStatus(cudaMemcpy(rhs_device, rhs_host.data(), rhs_host.size() * sizeof(float),
                           cudaMemcpyHostToDevice));

    const float expected = DotProductSpec(lhs_host, rhs_host);

    DotProduct(lhs_device, rhs_device, lhs_host.size(), workspace_device, out_device);

    float out_host = 0.0f;
    CheckStatus(cudaMemcpy(&out_host, out_device, sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(out_host == expected);

    CheckStatus(cudaFree(lhs_device));
    CheckStatus(cudaFree(rhs_device));
    CheckStatus(cudaFree(workspace_device));
    CheckStatus(cudaFree(out_device));
}

}  // namespace

TEST_CASE("DotProduct") {
    SECTION("Basic") {
        const auto [lhs, rhs] =
            GENERATE(std::make_pair(std::vector<float>{1.0f}, std::vector<float>{-1.0f}),
                     std::make_pair(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                                    std::vector<float>{-1.0f, -2.0f, -3.0f, -4.0f, -5.0f}));

        DoDotProductTest(lhs, rhs);
    }

    SECTION("Large") {
        std::mt19937 gen{42};
        std::uniform_int_distribution<int> distribution{-10, 10};

        std::vector<float> lhs(1'000'000);
        std::vector<float> rhs(1'000'000);
        for (auto& value : lhs) {
            value = distribution(gen);
        }
        for (auto& value : rhs) {
            value = distribution(gen);
        }

        DoDotProductTest(lhs, rhs);
    }
}
