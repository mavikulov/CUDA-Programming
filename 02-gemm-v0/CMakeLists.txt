add_catch(test_gemm_v0 test.cu gemm.cuh)
target_link_libraries(test_gemm_v0 PRIVATE ysda_cuda::util)
