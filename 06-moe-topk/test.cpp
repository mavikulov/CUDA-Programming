#include "moe_topk.cuh"

#include <cuda_helpers.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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

void MoeTopKSpec(size_t batchSize, size_t numExperts, size_t topK,
                 const std::vector<__half>& logits, size_t logits_stride,
                 std::vector<int32_t>& out_idx, size_t idx_stride, std::vector<__half>& out_w,
                 size_t w_stride) {
    std::vector<bool> used(numExperts, false);
    for (size_t t = 0; t < batchSize; ++t) {
        for (size_t k = 0; k < topK; ++k) {
            __half best_v = __float2half(-std::numeric_limits<float>::infinity());
            size_t best_e = numExperts;
            for (size_t e = 0; e < numExperts; ++e) {
                if (used[e]) {
                    continue;
                }
                const __half v = logits[t * logits_stride + e];
                if (v > best_v || (v == best_v && e < best_e)) {
                    best_v = v;
                    best_e = e;
                }
            }
            used[best_e] = true;
            out_idx[t * idx_stride + k] = static_cast<int32_t>(best_e);
            out_w[t * w_stride + k] = best_v;
        }
        std::fill(used.begin(), used.end(), false);
    }
}

void AssertMoeTopKMatchesSpec(size_t batchSize, size_t numExperts, size_t topK,
                              const std::vector<__half>& logits, size_t logitsStride,
                              size_t idxStride, size_t wStride) {
    REQUIRE(logitsStride >= numExperts);
    REQUIRE(idxStride >= topK);
    REQUIRE(wStride >= topK);
    REQUIRE(logits.size() >= batchSize * logitsStride);

    std::vector<int32_t> ref_idx(batchSize * idxStride);
    std::vector<__half> ref_w(batchSize * wStride);
    MoeTopKSpec(batchSize, numExperts, topK, logits, logitsStride, ref_idx, idxStride, ref_w,
                wStride);

    Matrix<__half> logits_dev = AllocDeviceMatrix<__half>(batchSize, logitsStride);
    CheckStatus(cudaMemcpy2D(reinterpret_cast<void*>(logits_dev.pointer), logits_dev.stride,
                             logits.data(), logitsStride * sizeof(__half),
                             numExperts * sizeof(__half), batchSize, cudaMemcpyHostToDevice));

    Matrix<int32_t> idx_dev = AllocDeviceMatrix<int32_t>(batchSize, idxStride);
    Matrix<__half> w_dev = AllocDeviceMatrix<__half>(batchSize, wStride);

    INFO("tokens=" << batchSize << " experts=" << numExperts << " topK=" << topK
                   << " log_stride=" << logitsStride << " idx_stride=" << idxStride);

    MoeTopK(batchSize, numExperts, topK, logits_dev.pointer, logits_dev.stride / sizeof(__half),
            idx_dev.pointer, idx_dev.stride / sizeof(int32_t), w_dev.pointer,
            w_dev.stride / sizeof(__half));
    CheckStatus(cudaGetLastError());

    std::vector<int32_t> gpu_idx(batchSize * idxStride);
    std::vector<__half> gpu_w(batchSize * wStride);
    CheckStatus(cudaMemcpy2D(reinterpret_cast<void*>(gpu_idx.data()), idxStride * sizeof(int32_t),
                             idx_dev.pointer, idx_dev.stride, topK * sizeof(int32_t), batchSize,
                             cudaMemcpyDeviceToHost));
    CheckStatus(cudaMemcpy2D(reinterpret_cast<void*>(gpu_w.data()), wStride * sizeof(__half),
                             w_dev.pointer, w_dev.stride, topK * sizeof(__half), batchSize,
                             cudaMemcpyDeviceToHost));

    for (size_t t = 0; t < batchSize; ++t) {
        for (size_t k = 0; k < topK; ++k) {
            CHECK(gpu_idx[t * idxStride + k] == ref_idx[t * idxStride + k]);
            REQUIRE_THAT(gpu_w[t * wStride + k],
                         Catch::Matchers::WithinAbs(ref_w[t * wStride + k], 5e-5f));
        }
    }

    CheckStatus(cudaFree(logits_dev.pointer));
    CheckStatus(cudaFree(idx_dev.pointer));
    CheckStatus(cudaFree(w_dev.pointer));
}

void DoMoeTopKTest(size_t batchSize, size_t numExperts, size_t topK) {
    std::mt19937_64 rng(9001);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

    std::vector<__half> logits(batchSize * numExperts);
    for (__half& x : logits) {
        x = dist(rng);
    }

    __half maxOnGen = __float2half(-std::numeric_limits<float>::infinity());
    size_t maxIdx = logits.size();
    for (size_t i = 0; i < logits.size(); i++) {
        if (logits[i] > maxOnGen) {
            maxOnGen = logits[i];
            maxIdx = i;
        }
    }
    if (maxIdx == logits.size() - 1) {
        logits[logits.size() / 2] = maxOnGen;
    } else {
        logits[logits.size() - 1] = maxOnGen;
    }
    AssertMoeTopKMatchesSpec(batchSize, numExperts, topK, logits, numExperts, topK, topK);
}

}  // namespace

TEST_CASE("MoeTopK") {
    SECTION("Basic") {
        const auto batchSize = GENERATE(1u, 3u, 16u, 31u);
        const auto numExperts = GENERATE(4u, 8u, 16u, 32u, 17u);
        const auto topK = GENERATE(1u, 2u, 4u);

        REQUIRE(topK <= numExperts);
        DoMoeTopKTest(batchSize, numExperts, topK);
    }

    SECTION("Large") {
        const auto batchSize = GENERATE(512u, 2048u);
        const auto numExperts = GENERATE(64u, 128u);
        const auto topK = GENERATE(2u, 4u, 8u);

        REQUIRE(topK <= numExperts);
        DoMoeTopKTest(batchSize, numExperts, topK);
    }
}

TEST_CASE("MoeTopK deterministic") {
    SECTION("All logits equal — tie-break by smaller index") {
        const size_t batch = 3;
        const size_t experts = 11;
        const size_t k = 5;
        std::vector<__half> logits(batch * experts, __float2half(0.25f));
        AssertMoeTopKMatchesSpec(batch, experts, k, logits, experts, k, k);
        std::vector<int32_t> ref_idx(batch * k);
        std::vector<__half> ref_w(batch * k);
        MoeTopKSpec(batch, experts, k, logits, experts, ref_idx, k, ref_w, k);
        for (size_t t = 0; t < batch; ++t) {
            for (size_t i = 0; i < k; ++i) {
                REQUIRE(ref_idx[t * k + i] == static_cast<int32_t>(i));
            }
        }
    }

    SECTION("Strictly decreasing per row") {
        const size_t batch = 2;
        const size_t experts = 9;
        const size_t k = 4;
        std::vector<__half> logits(batch * experts);
        for (size_t t = 0; t < batch; ++t) {
            for (size_t e = 0; e < experts; ++e) {
                logits[t * experts + e] = __float2half(100.0f - static_cast<float>(e));
            }
        }
        AssertMoeTopKMatchesSpec(batch, experts, k, logits, experts, k, k);
    }

    SECTION("Partial ties — duplicate maxima") {
        const size_t experts = 8;
        const size_t k = 4;
        std::vector<__half> logits(experts);
        logits[0] = logits[1] = __float2half(2.0f);
        logits[2] = logits[3] = __float2half(1.0f);
        for (size_t e = 4; e < experts; ++e) {
            logits[e] = __float2half(0.1f * static_cast<float>(e));
        }
        AssertMoeTopKMatchesSpec(1, experts, k, logits, experts, k, k);
    }
}

TEST_CASE("MoeTopK shapes and strides") {
    SECTION("Single expert, topK 1") {
        std::vector<__half> logits = {__float2half(-2.0f)};
        AssertMoeTopKMatchesSpec(1, 1, 1, logits, 1, 1, 1);
    }

    SECTION("topK equals numExperts") {
        const size_t experts = 6;
        std::vector<__half> logits(experts);
        for (size_t e = 0; e < experts; ++e) {
            logits[e] = __float2half(static_cast<float>(e) * 0.5f);
        }
        AssertMoeTopKMatchesSpec(1, experts, experts, logits, experts, experts, experts);
    }

    SECTION("Padded logits row stride") {
        const size_t experts = 29;
        const size_t log_stride = 64;
        const size_t batch = 3;
        const size_t k = 3;
        std::vector<__half> logits(batch * log_stride, __float2half(-999.0f));
        for (size_t t = 0; t < batch; ++t) {
            for (size_t e = 0; e < experts; ++e) {
                logits[t * log_stride + e] =
                    __float2half(static_cast<float>(e % 7) * 0.1f + static_cast<float>(t) * 0.01f);
            }
        }
        AssertMoeTopKMatchesSpec(batch, experts, k, logits, log_stride, k, k);
    }

    SECTION("Batch sizes across block tiling") {
        const size_t experts = 16;
        const size_t k = 2;
        for (size_t batch : {1u, 2u, 3u, 4u, 5u, 15u, 16u, 17u}) {
            std::vector<__half> logits(batch * experts);
            for (size_t i = 0; i < logits.size(); ++i) {
                logits[i] = __float2half(static_cast<float>(i % 31) * 0.03f - 1.0f);
            }
            AssertMoeTopKMatchesSpec(batch, experts, k, logits, experts, k, k);
        }
    }

    SECTION("numExperts not divisible by 8") {
        const size_t experts = 35;
        const size_t k = 7;
        const size_t batch = 4;
        std::vector<__half> logits(batch * experts);
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (__half& x : logits) {
            x = __float2half(dist(rng));
        }
        AssertMoeTopKMatchesSpec(batch, experts, k, logits, experts, k, k);
    }

    SECTION("Many experts") {
        const size_t experts = 256;
        const size_t batch = 2;
        const size_t k = 5;
        std::vector<__half> logits(batch * experts);
        std::mt19937_64 rng(123);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (__half& x : logits) {
            x = __float2half(dist(rng));
        }
        AssertMoeTopKMatchesSpec(batch, experts, k, logits, experts, k, k);
    }
}
