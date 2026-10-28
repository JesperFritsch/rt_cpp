cmake_minimum_required(VERSION 3.20)
project(rt_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RT_WERROR      "Treat warnings as errors" OFF)
option(RT_BUILD_TESTS "Build the unit tests"     ON)
option(RT_BUILD_APPS  "Build the demo programs"  ON)

# Threads are needed almost everywhere in this project.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# --------------------------------------------------------------------------
# rt_warnings - sourceless target carrying the warning policy.
# Linked PRIVATE by our targets; never applied to fetched dependencies.
# --------------------------------------------------------------------------
add_library(rt_warnings INTERFACE)

target_compile_options(rt_warnings INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wcast-align
    -Wold-style-cast
    -Wnon-virtual-dtor
    -Woverloaded-virtual
    -Wnull-dereference
    -Wimplicit-fallthrough
    -Wformat=2
    $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-cond>
    $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-branches>
    $<$<CXX_COMPILER_ID:GNU>:-Wlogical-op>
    $<$<CXX_COMPILER_ID:GNU>:-Wuseless-cast>
    $<$<BOOL:${RT_WERROR}>:-Werror>
)

# --------------------------------------------------------------------------
# rt_core - the library. Header-only for now; add sources to src/ as needed
# and switch INTERFACE to STATIC when the first .cpp appears.
# --------------------------------------------------------------------------
add_library(rt_core STATIC src/FDTDGrid.cpp)

target_include_directories(rt_core PUBLIC include)
target_compile_features(rt_core PUBLIC cxx_std_17)
target_link_libraries(rt_core
  PUBLIC Threads::Threads
  PRIVATE rt_warnings)
target_compile_options(rt_core PRIVATE $<$<CONFIG:Release>:-march=native>)

find_package(OpenMP REQUIRED)
target_link_libraries(rt_core PRIVATE OpenMP::OpenMP_CXX)

# --------------------------------------------------------------------------
if(RT_BUILD_APPS)
    add_subdirectory(apps)
endif()

if(RT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
