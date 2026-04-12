#pragma once

#include <cstdio>
#include <stdexcept>

#include <cuda_helpers.h>

__global__ void KernelHelloWorld() {
    printf("Hello, world!");
}

void CallHelloWorld() {
    KernelHelloWorld<<<1, 1>>>();
    cudaDeviceSynchronize();
}
