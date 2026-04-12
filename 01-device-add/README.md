# Device Add

В этой задаче вам предстоит написать код, складывающий два `float` массива на GPU.
В частности, помимо самого кернела, необходимо реализовать функции, выделяющие/освобождающие память на GPU, а также копирующие массивы с хоста на GPU и обратно:

```cpp
float* AllocDeviceVector(size_t num_elements) {
    // YOUR CODE HERE
}

void FreeDeviceVector(float* device_ptr) {
    // YOUR CODE HERE
}

void CopyHostVectorToDevice(const std::vector<float>& vector_host, float* dst_device_ptr) {
    // YOUR CODE HERE
}

std::vector<float> CopyDeviceVectorToHost(const float* ptr_device, size_t num_elements) {
    // YOUR CODE HERE
}

void AddDeviceVectors(const float* left_device, const float* right_device, float* out_device,
                      size_t num_elements) {
    // YOUR CODE HERE
}
```

Для проверки ошибок CUDA удобно использовать функции из `cuda_helpers.h`.

## Полезные ссылки

- [An easy introduction to CUDA C and C++](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)
- [Материалы второй лекции](https://lk.dataschool.yandex.ru/courses/2026-spring/7.1677-CUDA/classes/15413/)
