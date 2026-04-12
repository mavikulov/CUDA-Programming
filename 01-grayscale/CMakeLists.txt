add_catch(test_grayscale test.cpp grayscale.cuh grayscale.cu)
target_link_libraries(test_grayscale PRIVATE ysda_cuda::util stb::stb_image)
