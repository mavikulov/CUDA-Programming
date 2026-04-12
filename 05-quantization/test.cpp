#include "quantization.cuh"

#include <cuda_helpers.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

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

void QuantizeSpec(size_t rows, size_t cols, const std::vector<float>& input,
                  const std::vector<float>& balances, std::vector<int8_t>& out,
                  std::vector<float>& outputScales) {
    const float* inputRaw = input.data();
    int8_t* outputRaw = out.data();
    float* scalesRaw = outputScales.data();
    const float* smoothFactorsRaw = balances.data();

    for (size_t row = 0; row < rows; row++) {
        const float* inPos = inputRaw + row * cols;
        const float* inEndPos = inPos + cols;
        float localScale = 1e-5;
        for (size_t col = 0; col < cols; col++) {
            float absVal = std::abs(inPos[col] + smoothFactorsRaw[col]);
            if (absVal > localScale) {
                localScale = absVal;
            }
        }
        float scale = 127.f / localScale;
        int8_t* outPos = outputRaw + row * cols;
        std::transform(inPos, inEndPos, smoothFactorsRaw, outPos,
                       [scale](const float& val, const float& factor) -> int8_t {
                           return std::round((val + factor) * scale);
                       });
        scalesRaw[row] = scale;
    }
}

void DoQuantizationTest(size_t lineSize, size_t linesCount) {
    std::vector<float> inputMatrix(lineSize * linesCount);
    std::iota(inputMatrix.begin(), inputMatrix.end(), 0);

    std::vector<float> inputFactors(lineSize);
    std::iota(inputFactors.begin(), inputFactors.end(), 2);

    std::vector<float> refScales(linesCount);
    std::vector<int8_t> refMatrix(linesCount * lineSize);
    QuantizeSpec(linesCount, lineSize, inputMatrix, inputFactors, refMatrix, refScales);

    float* inputFactorsDevice = AllocDeviceVector<float>(lineSize);
    CheckStatus(cudaMemcpy(inputFactorsDevice, inputFactors.data(),
                           inputFactors.size() * sizeof(float), cudaMemcpyHostToDevice));

    Matrix<float> inputMatrixDevice = AllocDeviceMatrix<float>(linesCount, lineSize);

    CheckStatus(cudaMemcpy2D(reinterpret_cast<void*>(inputMatrixDevice.pointer),
                             inputMatrixDevice.stride, inputMatrix.data(), lineSize * sizeof(float),
                             lineSize * sizeof(float), linesCount, cudaMemcpyHostToDevice));

    Matrix<int8_t> outMatrixDevice = AllocDeviceMatrix<int8_t>(linesCount, lineSize);

    float* outputScalesDevice = AllocDeviceVector<float>(linesCount);
    INFO("ROWS = " << linesCount << " COLS = " << lineSize);

    Quantization(linesCount, lineSize, inputMatrixDevice.pointer, inputFactorsDevice,
                 inputMatrixDevice.stride / sizeof(float), outMatrixDevice.stride,
                 outMatrixDevice.pointer, outputScalesDevice);
    CheckStatus(cudaGetLastError());

    std::vector<float> outScales(linesCount);
    std::vector<int8_t> outMatrix(linesCount * lineSize);
    CheckStatus(cudaMemcpy2D(reinterpret_cast<void*>(outMatrix.data()), lineSize,
                             outMatrixDevice.pointer, outMatrixDevice.stride, lineSize, linesCount,
                             cudaMemcpyDeviceToHost));

    CheckStatus(cudaMemcpy(outScales.data(), outputScalesDevice, refScales.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));

    CHECK(outScales == refScales);
    CHECK(outMatrix == refMatrix);

    CheckStatus(cudaFree(inputFactorsDevice));
    CheckStatus(cudaFree(outputScalesDevice));
    CheckStatus(cudaFree(outMatrixDevice.pointer));
    CheckStatus(cudaFree(inputMatrixDevice.pointer));
}

}  // namespace

TEST_CASE("Quantization") {
    SECTION("Basic") {
        const auto lineSize = GENERATE(16, 64, 32, 128);
        const auto lines = GENERATE(2, 1, 3, 4, 8, 16, 7);

        DoQuantizationTest(lineSize, lines);
    }

    SECTION("Large") {
        const auto lineSize = GENERATE(1024, 2048, 4096, 8192);
        const auto lines = GENERATE(100, 64, 32, 57);
        DoQuantizationTest(lineSize, lines);
    }
}
