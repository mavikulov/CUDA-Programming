add_catch(test_reverse_string test.cu reverse.cuh)
target_link_libraries(test_reverse_string PRIVATE ysda_cuda::util)
