# MoE Top-K + гистограмма экспертов

Та же постановка, что в **04-moe-topk** (Top-K по логитам `__half` для каждого токена), плюс **гистограмма нагрузки экспертов**.

Массив `expertHistogram` длины `numExperts`: `expertHistogram[e]` — сколько раз эксперт `e` попал в выборку среди всех токенов (каждый токен даёт `topK` попаданий, возможны повторы одного эксперта у разных токенов).

Гарантируется, что `expertHistogram` занулена перед вызовом `MoeTopKHist`.

```cpp
void MoeTopKHist(size_t batchSize, size_t numExperts, size_t topK, const __half* logits,
                 size_t inputStride, int32_t* outIdxs, size_t idxsStride, __half* topkWeights,
                 size_t outStride, unsigned int* expertHistogram);
```

Логиты могут содержать **−∞**.
