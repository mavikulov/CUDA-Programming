
#include "quantization.cuh"
#include <cstddef>
#include <numeric>
#include <vector>

#include <cuda_helpers.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace {

template <typename T>
struct Matrix {
    size_t stride;
    T* pointer;
};

template <typename Value>
Value* AllocDeviceVector(size_t count) {
    Value* ptr_device = nullptr;

    CheckStatus(cudaMalloc(reinterpret_cast<void**>(&ptr_device), count * sizeof(Value)));

    return ptr_device;
}

template <typename T>
Matrix<T> AllocDeviceMatrix(size_t lines, size_t lineSize) {
    uint8_t* device_ptr = nullptr;
    size_t stride = 0;

    CheckStatus(cudaMallocPitch(reinterpret_cast<void**>(&device_ptr), &stride,
                                lineSize * sizeof(T), lines));

    return {.stride = stride, .pointer = reinterpret_cast<T*>(device_ptr)};
}

struct BenchmarkData {
    BenchmarkData(size_t lineSize, size_t linesCount)
        : LineSize_(lineSize), LinesCount_(linesCount) {
        std::vector<float> inputMatrix(lineSize * linesCount);
        std::iota(inputMatrix.begin(), inputMatrix.end(), 0);

        std::vector<float> inputFactors(lineSize);
        std::iota(inputFactors.begin(), inputFactors.end(), 2);

        InputFactorsDevice_ = AllocDeviceVector<float>(lineSize);
        CheckStatus(cudaMemcpy(InputFactorsDevice_, inputFactors.data(),
                               inputFactors.size() * sizeof(float), cudaMemcpyHostToDevice));

        InputMatrixDevice_ = AllocDeviceMatrix<float>(linesCount, lineSize);

        CheckStatus(cudaMemcpy2D(reinterpret_cast<void*>(InputMatrixDevice_.pointer),
                                 InputMatrixDevice_.stride, inputMatrix.data(),
                                 lineSize * sizeof(float), lineSize * sizeof(float), linesCount,
                                 cudaMemcpyHostToDevice));
        OutMatrixDevice_ = AllocDeviceMatrix<int8_t>(linesCount, lineSize);
        OutputScalesDevice_ = AllocDeviceVector<float>(linesCount);
    }
    void DoBenchmark() {
        Quantization(LinesCount_, LineSize_, InputMatrixDevice_.pointer, InputFactorsDevice_,
                     InputMatrixDevice_.stride / sizeof(float), OutMatrixDevice_.stride,
                     OutMatrixDevice_.pointer, OutputScalesDevice_);
        CheckStatus(cudaGetLastError());
        CheckStatus(cudaDeviceSynchronize());
    }
    ~BenchmarkData() {
        CheckStatus(cudaFree(InputFactorsDevice_));
        CheckStatus(cudaFree(OutputScalesDevice_));
        CheckStatus(cudaFree(OutMatrixDevice_.pointer));
        CheckStatus(cudaFree(InputMatrixDevice_.pointer));
    }

private:
    size_t LineSize_;
    size_t LinesCount_;
    float* InputFactorsDevice_;
    float* OutputScalesDevice_;
    Matrix<float> InputMatrixDevice_;
    Matrix<int8_t> OutMatrixDevice_;
};

}  // namespace

TEST_CASE("BenchmarkQuantization") {
    BenchmarkData large(8192, 4000);
    BenchmarkData small(3072, 8);

    BENCHMARK("QuantizationBigKernel") {
        large.DoBenchmark();
    };
    BENCHMARK("QuantizationSmallKernel") {
        small.DoBenchmark();
    };
}
