add_catch(test_transpose_v0 test.cu transpose.cuh)
target_link_libraries(test_transpose_v0 PRIVATE ysda_cuda::util)
