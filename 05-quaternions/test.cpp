#include "quaternions.cuh"
#include "quaternions_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_helpers.h>
#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

namespace {
class QuaternionsEqualsMatcher : public Catch::Matchers::MatcherGenericBase {
public:
    QuaternionsEqualsMatcher(const std::vector<Quaternion>& vec) : vec_{vec} {
    }

    bool match(const std::vector<Quaternion>& other) const {  // NOLINT
        if (vec_.size() != other.size()) {
            return false;
        }

        auto cmp_floats = [](float ref, float val) -> bool {
            return std::abs(val - ref) < 1e-5 * std::max(1.0f, ref);
        };

        for (size_t i = 0; i < vec_.size(); ++i) {
            const auto& ref_quat = vec_[i];
            const auto& quat = other[i];
            if (!cmp_floats(ref_quat.a, quat.a) || !cmp_floats(ref_quat.b, quat.b) ||
                !cmp_floats(ref_quat.c, quat.c) || !cmp_floats(ref_quat.d, quat.d)) {
                return false;
            }
        }

        return true;
    }

    std::string describe() const override {
        return "QuaternionEquals: " + Catch::rangeToString(vec_);
    }

private:
    const std::vector<Quaternion>& vec_;
};

void RefQuaternionReduce(size_t rows, size_t cols, const Quaternion* inp, Quaternion* out) {
    QuaternionMultiplier multiplier{};

    for (size_t row_idx = 0; row_idx < rows; ++row_idx, inp += cols, ++out) {
        Quaternion res = inp[0];

        for (size_t col_idx = 1; col_idx < cols; ++col_idx) {
            res = multiplier(res, inp[col_idx]);
        }

        *out = res;
    }
}

void DoQuaternionTest(size_t line_size, size_t lines_count, float std_dev, size_t seed_value) {
    std::seed_seq seed{seed_value};
    std::mt19937 mt{seed};
    std::normal_distribution<> normal_dist{};

    auto gen_normal_float = [&]() -> float { return normal_dist(mt) * std_dev; };

    std::vector<Quaternion> input_matrix(lines_count * line_size);
    for (size_t line_idx = 0; line_idx < lines_count; ++line_idx) {
        for (size_t in_line_idx = 0; in_line_idx < line_size; ++in_line_idx) {
            input_matrix[line_idx * line_size + in_line_idx] =
                quaternions::GenQuaternion(gen_normal_float);
        }
    }

    std::vector<Quaternion> out_matrix_ref(lines_count);
    RefQuaternionReduce(lines_count, line_size, input_matrix.data(), out_matrix_ref.data());

    cudaStream_t stream;
    CheckStatus(cudaStreamCreate(&stream));

    auto input_matrix_device = quaternions::AllocDeviceMatrix<Quaternion>(lines_count, line_size);
    CheckStatus(cudaMemcpy2D(reinterpret_cast<void*>(input_matrix_device.pointer),
                             input_matrix_device.stride * sizeof(Quaternion), input_matrix.data(),
                             line_size * sizeof(Quaternion), line_size * sizeof(Quaternion),
                             lines_count, cudaMemcpyHostToDevice));

    auto out_vector_device = quaternions::AllocDeviceVector<Quaternion>(lines_count);

    INFO("ROWS = " << lines_count << " COLS = " << line_size << " STD_DEV = " << std_dev
                   << " SEED = " << seed_value);

    QuaternionsReduce(lines_count, line_size, input_matrix_device.pointer,
                      input_matrix_device.stride, out_vector_device, stream);

    CheckStatus(cudaGetLastError());
    CheckStatus(cudaStreamSynchronize(stream));

    std::vector<Quaternion> out_matrix(lines_count);
    CheckStatus(cudaMemcpy(out_matrix.data(), reinterpret_cast<void*>(out_vector_device),
                           lines_count * sizeof(Quaternion), cudaMemcpyDeviceToHost));

    REQUIRE_THAT(out_matrix, QuaternionsEqualsMatcher(out_matrix_ref));

    CheckStatus(cudaFree(input_matrix_device.pointer));
    CheckStatus(cudaFree(out_vector_device));
    CheckStatus(cudaStreamDestroy(stream));
}
}  // namespace

TEST_CASE("Softmax") {
    SECTION("Large") {
        const auto line_size = GENERATE(1024, 2048, 4096);
        const auto lines_count = GENERATE(1, 4, 47, 100);
        DoQuaternionTest(line_size, lines_count, 20, 42);
    }
}
