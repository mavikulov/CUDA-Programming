add_catch(test_moe_topk_hist test.cpp moe_topk_hist.cu)
add_catch(bench_moe_topk_hist run.cpp moe_topk_hist.cu)
target_compile_options(test_moe_topk_hist PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(bench_moe_topk_hist PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_link_libraries(test_moe_topk_hist PRIVATE ysda_cuda::util)
target_link_libraries(bench_moe_topk_hist PRIVATE ysda_cuda::util)
set_target_properties(test_moe_topk_hist PROPERTIES CUDA_ARCHITECTURES "50;75;86")
set_target_properties(bench_moe_topk_hist PROPERTIES CUDA_ARCHITECTURES "50;75;86")
