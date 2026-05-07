# Prefix sum

В этой задаче вам необходимо реализовать префиксное суммирование (кумулятивное суммирование, exclusive scan) массива размера $N$:
$$
b_0 = 0 \\
b_j = \sum_{i=0}^{j-1}a_i
$$

Например: $\left[1, 2, 3\right] \rightarrow [0, 1, 3, 6]$.

Гарантируется, что $N \leq 10^6$.
В качестве типа данных необходимо использовать тип int, при этом гарантируется, что при суммировании не будет переполнения.

## Полезные ссылки
- [Префиксная сумма - Википедия](https://ru.wikipedia.org/wiki/%D0%9F%D1%80%D0%B5%D1%84%D0%B8%D0%BA%D1%81%D0%BD%D0%B0%D1%8F_%D1%81%D1%83%D0%BC%D0%BC%D0%B0)
- [Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
