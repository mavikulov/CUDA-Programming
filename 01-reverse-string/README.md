# Reverse String

В этой задаче вам предстоит написать код, переворачивающий строку размера $N$ ($N \leq 10^{10}$) на GPU in-place, то есть без выделения дополнительной памяти под результат:

$$
roza \rightarrow azor
$$

```cpp
void ReverseDeviceStringInplace(char* str, size_t length);  // Результат нужно записать прямо в str
```

**Важно:** в данной задаче есть тесты на достаточно большом объёме данных (превышающем 8 ГБ).
В случае попытки решения задачи на imladris на картах RTX 4000 доступной видеопамяти может не хватить, поэтому для решения этой задачи необходимо использовать RTX A4000 с 16 ГБ видеопамяти.
На beleriand такой проблемы нет, поскольку там все карты - RTX A4000.
Список доступных GPU и объём свободной памяти можно узнать командой `nvidia-smi`, а затем указать тесту, на какой GPU запускаться, переменной окружения `CUDA_VISIBLE_DEVICES`:

```bash
cd build
ninja test_reverse_string
CUDA_VISIBLE_DEVICES=2 ./test_reverse_string
```

Однако нумерация GPU, используемая в nvidia-smi, может не совпадать с нумерацией внутри `CUDA_VISIBLE_DEVICES` из-за настроек драйвера.
Эту проблему можно решить, узнав UUID конкретного GPU через `nvidia-smi -L`, а затем передав его в `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=GPU-... ./test_reverse_string
```

Также порядок GPU внутри `CUDA_VISIBLE_DEVICES` можно регулировать через [`CUDA_DEVICE_ORDER`](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-order).

Для удобства отладки тесты разделены на две группы: с маленькими и большими данными.
