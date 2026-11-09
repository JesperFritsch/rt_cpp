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
# 4.3 is the first version with shader storage buffers and compute shaders
set(OPENGL_VERSION "4.3" CACHE STRING "OpenGL version raylib is built for" FORCE)
FetchContent_Declare(
  raylib
  GIT_REPOSITORY https://github.com/raysan5/raylib.git
  GIT_TAG 5.5 SYSTEM)
FetchContent_MakeAvailable(raylib)
target_compile_options(raylib PRIVATE -w)

# the GPU backend needs raylib's OpenGL context, so it lives apart from rt_core
add_library(rt_gpu STATIC ${PROJECT_SOURCE_DIR}/src/gpu_backend.cpp)
target_link_libraries(rt_gpu PUBLIC rt_core raylib PRIVATE rt_warnings)

add_executable(radar_sim radar_sim.cpp)
target_link_libraries(radar_sim PRIVATE rt_radar rt_gpu rt_warnings raylib)

# checks the GPU backend against the CPU one and times both
add_executable(backend_compare backend_compare.cpp)
target_link_libraries(backend_compare PRIVATE rt_core rt_gpu rt_warnings raylib)
