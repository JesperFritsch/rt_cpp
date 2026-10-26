add_executable(demo demo.cpp)
target_link_libraries(demo PRIVATE rt_core rt_warnings)

add_executable(ring_demo spsc_demo.cpp)
target_link_libraries(ring_demo PRIVATE rt_core rt_warnings)

add_executable(ring_bench bench_spsc.cpp)
target_link_libraries(ring_bench PRIVATE rt_core rt_warnings)

add_executable(single_ring_bench single_t_ring_bench.cpp)
target_link_libraries(single_ring_bench PRIVATE rt_core rt_warnings)

add_executable(lgr_bench bench_logger.cpp)
target_link_libraries(lgr_bench PRIVATE rt_core rt_warnings)

include(FetchContent)
FetchContent_Declare(
  raylib
  GIT_REPOSITORY https://github.com/raysan5/raylib.git
  GIT_TAG 5.5 SYSTEM)
FetchContent_MakeAvailable(raylib)
target_compile_options(raylib PRIVATE -w)

add_executable(radar_sim radar_sim.cpp)
target_link_libraries(radar_sim PRIVATE rt_core rt_warnings raylib)
