#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

// Represents quaternion a*1 + b*i + c*j + d*k
struct alignas(16) Quaternion {
    float a = 0.f;
    float b = 0.f;
    float c = 0.f;
    float d = 0.f;
};

// https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
struct QuaternionMultiplier {
    __host__ __device__ __forceinline__ Quaternion operator()(const Quaternion& l,
                                                              const Quaternion& r) const {
        return {
            //   a_1 * a_2 - b_1 * b_2 - c_1 * c_2 - d_1 * d_2
            .a = l.a * r.a - l.b * r.b - l.c * r.c - l.d * r.d,
            // ( a_1 * b_2 + b_1 * a_2 + c_1 * d_2 - d_1 * c_2 ) * i
            .b = l.a * r.b + l.b * r.a + l.c * r.d - l.d * r.c,
            // ( a_1 * c_2 - b_1 * d_2 + c_1 * a_2 + d_1 * b_2 ) * j
            .c = l.a * r.c - l.b * r.d + l.c * r.a + l.d * r.b,
            // ( a_1 * d_2 + b_1 * c_2 - c_1 * b_2 + d_1 * a_2 ) * k
            .d = l.a * r.d + l.b * r.c - l.c * r.b + l.d * r.a,
        };
    }
};

void QuaternionsReduce(size_t rows, size_t cols, const Quaternion* inp, size_t inp_stride,
                       Quaternion* out, cudaStream_t stream);
