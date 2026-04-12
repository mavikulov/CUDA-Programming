# Transpose v0

В данной задаче вам предстоит реализовать операцию транспонирования матриц через shared memory:

$$
\begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{pmatrix}^{T}=
\begin{pmatrix}
1 & 4 \\
2 & 5 \\
3 & 6
\end{pmatrix}
$$

В отличие от предыдущей задачи, в данной задаче все матрицы предполагают row-major укладку в памяти.
В данной версии задачи также отсутствуют бенчмарки, но в дальнейшем будут усложненные версии этой задачи, где бенчмарки уже будут.

## Полезные ссылки
- [Using shared memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
