# Grayscale

В этой задаче вам предстоит написать функцию, переводящую цветное трёхканальное RGB изображение в одноканальное чёрно-белое на GPU.
Изображение представляет из себя двумерный массив формата HWC, где подряд пишутся значения красного $R$, зеленого $G$ и синего $B$ каналов для каждого пикселя:
$$
\begin{matrix}
    R_{00} & G_{00} & B_{00} & R_{01} & G_{01} & B_{01} \\
    R_{10} & G_{10} & B_{10} & R_{11} & G_{11} & B_{11}
\end{matrix}
$$
В качестве формулы преобразования каналов предлагается использовать следующую:
$$
Y = 0.299 \times R + 0.587 \times G + 0.114 \times B
$$

Помимо самой функции преобразования также предлагается написать несколько вспомогательных функций, отвечающих за выделение памяти и копирование изображения на GPU и обратно:

```cpp
// grayscale.cuh
Image AllocHostImage(size_t width, size_t height, size_t channels);
Image AllocDeviceImage(size_t width, size_t height, size_t channels);
void CopyImageHostToDevice(const Image& src_host, Image& dst_device);
void CopyImageDeviceToHost(const Image& src_device, Image& dst_host);
void ConvertToGrayscaleDevice(const Image& rgb_device_image, Image& gray_device_image);
void FreeDeviceImage(const Image& image);
void FreeHostImage(const Image& image);
```

Сама функция `ConvertToGrayscaleDevice` принимает вторым параметром уже выделенное выходное изображение.
Никаких аллокаций внутри неё делать не нужно.
Помимо этого, функция предполагается асинхронной, поэтому явных синхронизаций в её конце делать тоже не нужно.

Обратите внимание, что cтроки изображения могут быть отделены друг от друга в памяти промежутками с целью выравнивания, см. раздел Device Memory в CUDA C Programming Guide.
В данной задаче под параметром `stride` мы понимаем расстояние в байтах между началами соседних строк изображения.
В CUDA API данный параметр часто называется `pitch`, см. функции `cudaMallocPitch` и `cudaMemcpy2D`, которые могут пригодиться в этой задаче.

## Полезные ссылки

- [OpenCV: Color conversions](https://docs.opencv.org/4.10.0/de/d25/imgproc_color_conversions.html)
- [CUDA Toolkit Documentation: Memory management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY)
- [How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
