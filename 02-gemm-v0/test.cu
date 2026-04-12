#include "gemm.cuh"

#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace {

struct HostMatrix {
    HostMatrix(const std::vector<std::vector<__half>>& data2d, MatrixLayout layout)
        : layout{layout} {
        if (layout == MatrixLayout::RowMajor) {
            rows = data2d.size();
            cols = data2d[0].size();
        } else {
            cols = data2d.size();
            rows = data2d[0].size();
        }

        this->data.reserve(rows * cols);
        for (const auto& memory_row : data2d) {
            this->data.insert(this->data.end(), memory_row.begin(), memory_row.end());
        }
    }

    HostMatrix(std::vector<__half> data, size_t rows, size_t cols, MatrixLayout layout)
        : data(std::move(data)), rows{rows}, cols{cols}, layout{layout} {
    }

    DeviceMatrix ToGPU() const {
        __half* device_ptr = nullptr;
        const size_t width = layout == MatrixLayout::RowMajor ? cols : rows;
        const size_t height = layout == MatrixLayout::RowMajor ? rows : cols;
        size_t device_pitch = 0;
        CheckStatus(cudaMallocPitch(&device_ptr, &device_pitch, width * sizeof(__half), height));
        const size_t host_pitch = width * sizeof(__half);
        CheckStatus(cudaMemcpy2D(device_ptr, device_pitch, data.data(), host_pitch,
                                 width * sizeof(__half), height, cudaMemcpyHostToDevice));

        return DeviceMatrix{.data = device_ptr,
                            .rows = rows,
                            .cols = cols,
                            .stride = device_pitch / sizeof(__half),
                            .layout = layout};
    }

    static HostMatrix FromGPU(const DeviceMatrix& device_matrix) {
        std::vector<__half> data(device_matrix.rows * device_matrix.cols);
        const size_t width = device_matrix.layout == MatrixLayout::RowMajor ? device_matrix.cols
                                                                            : device_matrix.rows;
        const size_t height = device_matrix.layout == MatrixLayout::RowMajor ? device_matrix.rows
                                                                             : device_matrix.cols;
        const size_t host_pitch = width * sizeof(__half);
        CheckStatus(cudaMemcpy2D(data.data(), host_pitch, device_matrix.data,
                                 device_matrix.stride * sizeof(__half), width * sizeof(__half),
                                 height, cudaMemcpyDeviceToHost));

        return HostMatrix(std::move(data), device_matrix.rows, device_matrix.cols,
                          device_matrix.layout);
    }

    std::vector<__half> data;
    size_t rows;
    size_t cols;
    MatrixLayout layout;
};

DeviceMatrix AllocAlike(const DeviceMatrix& other) {
    __half* device_ptr;
    const size_t width = other.layout == MatrixLayout::RowMajor ? other.cols : other.rows;
    const size_t height = other.layout == MatrixLayout::RowMajor ? other.rows : other.cols;
    size_t device_pitch = 0;
    CheckStatus(cudaMallocPitch(&device_ptr, &device_pitch, width * sizeof(__half), height));

    return DeviceMatrix{.data = device_ptr,
                        .rows = other.rows,
                        .cols = other.cols,
                        .stride = device_pitch / sizeof(__half),
                        .layout = other.layout};
}

struct TestCase {
    HostMatrix a;
    HostMatrix b;
    HostMatrix c;
    HostMatrix expected;
    bool inplace;
    float alpha;
    float beta;
};

template <typename Generator>
HostMatrix GenerateRandomMatrix(size_t rows, size_t cols, MatrixLayout layout, Generator& gen) {
    std::vector<__half> values;
    const size_t num_elements = rows * cols;
    values.reserve(num_elements);

    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (size_t index = 0; index < num_elements; ++index) {
        values.push_back(static_cast<__half>(distribution(gen)));
    }

    return HostMatrix(std::move(values), rows, cols, layout);
}

TestCase GenerateRandomTestCase(size_t m, size_t k, size_t n, MatrixLayout layout_a,
                                MatrixLayout layout_b, MatrixLayout layout_c, bool inplace,
                                float alpha, float beta) {
    static std::mt19937 generator(42);

    HostMatrix a = GenerateRandomMatrix(m, k, layout_a, generator);
    HostMatrix b = GenerateRandomMatrix(k, n, layout_b, generator);
    HostMatrix c = GenerateRandomMatrix(m, n, layout_c, generator);
    HostMatrix d = c;

    for (size_t row = 0; row < d.rows; ++row) {
        for (size_t col = 0; col < d.cols; ++col) {
            auto get_value = [](HostMatrix& matrix, size_t row, size_t col) -> __half& {
                const size_t offset = matrix.layout == MatrixLayout::RowMajor
                                          ? row * matrix.cols + col
                                          : col * matrix.rows + row;
                return matrix.data[offset];
            };
            float accumulator = 0;
            for (size_t inner = 0; inner < k; ++inner) {
                accumulator +=
                    static_cast<float>(get_value(a, row, inner) * get_value(b, inner, col));
            }

            get_value(d, row, col) =
                alpha * accumulator + beta * static_cast<float>(get_value(c, row, col));
        }
    }

    return TestCase{
        .a = std::move(a),
        .b = std::move(b),
        .c = std::move(c),
        .expected = std::move(d),
        .inplace = inplace,
        .alpha = alpha,
        .beta = beta,
    };
}

}  // namespace

TEST_CASE("GEMM") {
    const auto test_case = GENERATE(
        TestCase{.a = {{{2.0f}}, MatrixLayout::RowMajor},
                 .b = {{{3.0f}}, MatrixLayout::ColMajor},
                 .c = {{{1.0f}}, MatrixLayout::RowMajor},
                 .expected = {{{7.0f}}, MatrixLayout::RowMajor},
                 .inplace = false,
                 .alpha = 1.0f,
                 .beta = 1.0f},
        TestCase{.a = {{{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}}, MatrixLayout::RowMajor},
                 .b = {{{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}}, MatrixLayout::ColMajor},
                 .c = {{{1.0f}}, MatrixLayout::ColMajor},
                 .expected = {{{29.5f}}, MatrixLayout::ColMajor},
                 .inplace = true,
                 .alpha = 0.5f,
                 .beta = 2.0f},
        TestCase{.a = {{{1.0f, 2.0f}, {-1.0f, -2.0f}}, MatrixLayout::ColMajor},
                 .b = {{{-3.0f, 4.0f}, {5.0f, 6.0f}}, MatrixLayout::ColMajor},
                 .c = {{{0.0f, 0.0f}, {0.0f, 0.0f}}, MatrixLayout::ColMajor},
                 .expected = {{{-7.0f, -14.0f}, {-1.0f, -2.0f}}, MatrixLayout::ColMajor},
                 .inplace = false,
                 .alpha = 1.0f,
                 .beta = 0.0f},
        GenerateRandomTestCase(30, 30, 30, MatrixLayout::RowMajor, MatrixLayout::ColMajor,
                               MatrixLayout::ColMajor, false, 2.0f, 1.0f),
        GenerateRandomTestCase(200, 80, 200, MatrixLayout::ColMajor, MatrixLayout::ColMajor,
                               MatrixLayout::RowMajor, false, -2.0f, 0.5f));

    DeviceMatrix a_device = test_case.a.ToGPU();
    DeviceMatrix b_device = test_case.b.ToGPU();
    DeviceMatrix c_device = test_case.c.ToGPU();
    DeviceMatrix d_device = test_case.inplace ? c_device : AllocAlike(c_device);

    REQUIRE(a_device.cols == b_device.rows);
    REQUIRE((c_device.rows == a_device.rows && c_device.cols == b_device.cols));
    REQUIRE((c_device.rows == d_device.rows && c_device.cols == d_device.cols));
    REQUIRE(c_device.layout == d_device.layout);

    GEMM(a_device, b_device, c_device, d_device, test_case.alpha, test_case.beta);
    CheckStatus(cudaGetLastError());

    HostMatrix out = HostMatrix::FromGPU(d_device);
    for (size_t row = 0; row < out.rows; ++row) {
        for (size_t col = 0; col < out.cols; ++col) {
            const size_t index =
                out.layout == MatrixLayout::RowMajor ? row * out.cols + col : col * out.rows + row;
            const float actual = out.data[index];
            const float expected = test_case.expected.data[index];
            INFO("row = " << row << " col = " << col);
            CHECK_THAT(actual, Catch::Matchers::WithinRel(expected, 1e-4f));
        }
    }

    CheckStatus(cudaFree(a_device.data));
    CheckStatus(cudaFree(b_device.data));
    CheckStatus(cudaFree(c_device.data));
    if (!test_case.inplace) {
        CheckStatus(cudaFree(d_device.data));
    }
}
