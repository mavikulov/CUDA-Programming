add_catch(test_device_add test.cu add.cuh)
target_link_libraries(test_device_add PRIVATE ysda_cuda::util)
