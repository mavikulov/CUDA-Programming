#include "reverse.cuh"

#include <algorithm>
#include <optional>
#include <random>
#include <string>
#include <thread>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

namespace {

struct TestCase {
    std::string input;
    std::string expected;
};

std::string ReverseStringSpec(std::string str) {
    const size_t length = str.size();
    char* str_ptr = str.data();
    for (size_t i = 0; i < (length - i - 1); ++i) {
        std::swap(str_ptr[i], str_ptr[length - i - 1]);
    }
    return str;
}

TestCase GenerateHugeTestCase(size_t length) {
    std::string result(length, ' ');

    std::vector<std::thread> workers;
    const size_t num_workers = std::min(8ul, length);
    const size_t chunk_size = (length + num_workers - 1) / num_workers;
    for (size_t worker_idx = 0; worker_idx < num_workers; ++worker_idx) {
        const size_t start = worker_idx * chunk_size;
        const size_t end = std::min((worker_idx + 1) * chunk_size, length);
        workers.emplace_back([&, start, end]() {
            std::mt19937 gen(42 * worker_idx);
            std::uniform_int_distribution distr{0, 25};
            for (size_t index = start; index < end; ++index) {
                result[index] = 'a' + distr(gen);
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    return {result, ReverseStringSpec(result)};
}

std::optional<size_t> FindFirstNotEqualChar(const std::string& lhs, const std::string& rhs) {
    REQUIRE(lhs.size() == rhs.size());

    for (size_t index = 0; index < lhs.size(); ++index) {
        if (lhs[index] != rhs[index]) {
            return index;
        }
    }

    return std::nullopt;
}

void DoTest(const TestCase& test_case) {
    const size_t length = test_case.input.size();
    char* str_device = nullptr;

    CheckStatus(cudaMalloc(&str_device, length));
    CheckStatus(cudaMemcpy(str_device, test_case.input.data(), length, cudaMemcpyHostToDevice));

    ReverseDeviceStringInplace(str_device, length);
    CheckStatus(cudaGetLastError());

    std::string out(length, ' ');
    CheckStatus(cudaMemcpy(out.data(), str_device, length, cudaMemcpyDeviceToHost));
    CheckStatus(cudaFree(str_device));

    if (length <= 1'000'000UL) {
        REQUIRE_THAT(out, Catch::Matchers::Equals(test_case.expected));
    } else {
        std::optional<size_t> neq_index = FindFirstNotEqualChar(out, test_case.expected);
        if (neq_index.has_value()) {
            FAIL("out and expected differ at index " << *neq_index << ", " << out[*neq_index]
                                                     << " != " << test_case.expected[*neq_index]);
        }
    }
}

}  // namespace

TEST_CASE("ReverseSimple") {
    const auto test_case =
        GENERATE(TestCase{"", ""}, TestCase{"a", "a"}, TestCase{"ab", "ba"}, TestCase{"aba", "aba"},
                 TestCase{"You probably recall over the course of the last 10-15 years, almost "
                          "everybody who sits on a stage like this would tell you it is vital that "
                          "your children learn computer science. Everybody should learn how to "
                          "program. In fact, it's almost exactly the opposite.",
                          ".etisoppo eht yltcaxe tsomla s'ti ,tcaf nI .margorp ot woh nrael dluohs "
                          "ydobyrevE .ecneics retupmoc nrael nerdlihc ruoy taht lativ si ti uoy "
                          "llet dluow siht ekil egats a no stis ohw ydobyreve tsomla ,sraey 51-01 "
                          "tsal eht fo esruoc eht revo llacer ylbaborp uoY"});

    DoTest(test_case);
}

TEST_CASE("ReverseHardcore") {
    const auto test_case =
        GENERATE(GenerateHugeTestCase(1'000'000), GenerateHugeTestCase(10'000'000'000UL));

    DoTest(test_case);
}
