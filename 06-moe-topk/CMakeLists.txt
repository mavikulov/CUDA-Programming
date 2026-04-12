add_catch(test_moe_topk test.cpp moe_topk.cu)
add_catch(bench_moe_topk run.cpp moe_topk.cu)
target_compile_options(test_moe_topk PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(bench_moe_topk PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_link_libraries(test_moe_topk PRIVATE ysda_cuda::util)
target_link_libraries(bench_moe_topk PRIVATE ysda_cuda::util)
set_target_properties(test_moe_topk PROPERTIES CUDA_ARCHITECTURES "50;75;86")
set_target_properties(bench_moe_topk PROPERTIES CUDA_ARCHITECTURES "50;75;86")