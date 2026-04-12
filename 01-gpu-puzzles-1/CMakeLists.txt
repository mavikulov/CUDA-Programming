add_catch(test_gpu_puzzles_1 test.cu puzzle.cuh)
target_link_libraries(test_gpu_puzzles_1 PRIVATE ysda_cuda::util)
