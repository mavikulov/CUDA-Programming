#pragma once

#include "quaternions.cuh"

#include <cmath>
#include <ostream>

#include <cuda_helpers.h>
#include <cuda_runtime.h>

namespace quaternions {
template <typename T>
struct Matrix {
    size_t stride;
    T* pointer;
};

template <typename T>
T* AllocDeviceVector(size_t count) {
    T* ptr_device = nullptr;
    CheckStatus(cudaMalloc(reinterpret_cast<void**>(&ptr_device), count * sizeof(T)));
    return ptr_device;
}

template <typename T>
Matrix<T> AllocDeviceMatrix(size_t lines, size_t line_size) {
    uint8_t* device_ptr = nullptr;
    size_t stride = 0;
    CheckStatus(cudaMallocPitch(reinterpret_cast<void**>(&device_ptr), &stride,
                                line_size * sizeof(T), lines));
    return {.stride = stride / sizeof(T), .pointer = reinterpret_cast<T*>(device_ptr)};
}

// Generate normalized quaternion
// so that during reduce we won't have problems with magnitude exploding / vanishing
template <typename Generator>
Quaternion GenQuaternion(Generator gen) {
    while (true) {
        float a = gen();
        float b = gen();
        float c = gen();
        float d = gen();

        float norm = std::sqrt(a * a + b * b + c * c + d * d);
        if (norm < 1e-5) {
            continue;
        }

        return {
            .a = a / norm,
            .b = b / norm,
            .c = c / norm,
            .d = d / norm,
        };
    }
}
}  // namespace quaternions

std::ostream& operator<<(std::ostream& os, const Quaternion& q) {
    return os << "(" << q.a << " + " << q.b << " * i + " << q.c << " * j + " << q.d << " * k)";
}
