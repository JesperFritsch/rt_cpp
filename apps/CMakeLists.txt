add_executable(demo demo.cpp)
target_link_libraries(demo PRIVATE rt_core rt_warnings)

add_executable(ring_demo spsc_demo.cpp)
target_link_libraries(ring_demo PRIVATE rt_core rt_warnings)

add_executable(ring_bench bench_spsc.cpp)
target_link_libraries(ring_bench PRIVATE rt_core rt_warnings)

add_executable(single_ring_bench single_t_ring_bench.cpp)
target_link_libraries(single_ring_bench PRIVATE rt_core rt_warnings)
