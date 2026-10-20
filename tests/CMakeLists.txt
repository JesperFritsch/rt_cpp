include(FetchContent)

FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE
)

set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_executable(rt_tests
    test_scaffold.cpp
    test_spsc_ring.cpp
    test_mpsc_ring.cpp
    test_logger.cpp
)

target_link_libraries(rt_tests
    PRIVATE
        rt_core
        rt_warnings
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(rt_tests PROPERTIES TIMEOUT 5)
