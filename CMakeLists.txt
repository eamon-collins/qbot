cmake_minimum_required(VERSION 3.20)
project(quoridor_mcts CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings (always enabled)
add_compile_options(-Wall -Wextra -Wpedantic)

# Find Boost program_options
find_package(Boost REQUIRED COMPONENTS program_options)

# =============================================================================
# Core Library (compiled separately for each build type)
# =============================================================================

set(QBOT_LIB_SOURCES
    src/tree/StateNode.cpp
    src/util/storage.cpp
    src/core/Game.cpp
)

set(QBOT_MAIN_SOURCES
    src/core/QuoridorMain.cpp
)

# =============================================================================
# Debug Build: "qbot" target (default)
# Compiles with debug symbols (-g -O0)
# =============================================================================

add_library(qbot_lib_debug STATIC ${QBOT_LIB_SOURCES})
target_include_directories(qbot_lib_debug PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(qbot_lib_debug PRIVATE -g -O0)

add_executable(qbot_exe_debug ${QBOT_MAIN_SOURCES})
target_link_libraries(qbot_exe_debug PRIVATE qbot_lib_debug Boost::program_options)
target_compile_options(qbot_exe_debug PRIVATE -g -O0)
set_target_properties(qbot_exe_debug PROPERTIES
    OUTPUT_NAME qbot
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# "qbot" target - builds debug version (this is the default)
add_custom_target(qbot DEPENDS qbot_exe_debug)

# =============================================================================
# Optimized Build: "fast" target
# Compiles with full optimizations (-O3 -march=native -flto)
# =============================================================================

add_library(qbot_lib_fast STATIC ${QBOT_LIB_SOURCES})
target_include_directories(qbot_lib_fast PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(qbot_lib_fast PRIVATE -O3 -DNDEBUG -march=native -flto)

add_executable(qbot_exe_fast ${QBOT_MAIN_SOURCES})
target_link_libraries(qbot_exe_fast PRIVATE qbot_lib_fast Boost::program_options)
target_compile_options(qbot_exe_fast PRIVATE -O3 -DNDEBUG -march=native -flto)
target_link_options(qbot_exe_fast PRIVATE -flto)
set_target_properties(qbot_exe_fast PROPERTIES
    OUTPUT_NAME qbot_fast_tmp
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# "fast" target - builds optimized version and overwrites the qbot binary
add_custom_target(fast
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/qbot_fast_tmp ${CMAKE_BINARY_DIR}/qbot
    DEPENDS qbot_exe_fast
    COMMENT "Building optimized qbot binary with -O3 -march=native -flto"
)

# =============================================================================
# Default target: build qbot (debug) when running just "make"
# =============================================================================

# Make qbot part of the default "all" target
add_dependencies(qbot_exe_debug qbot_lib_debug)

# =============================================================================
# Testing
# =============================================================================

option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(test_storage tests/test_storage.cpp)
    target_link_libraries(test_storage PRIVATE qbot_lib_debug GTest::gtest_main)
    target_compile_options(test_storage PRIVATE -g -O0)

    add_executable(test_game tests/test_game.cpp)
    target_link_libraries(test_game PRIVATE qbot_lib_debug GTest::gtest_main)
    target_compile_options(test_game PRIVATE -g -O0)

    include(GoogleTest)
    gtest_discover_tests(test_storage)
    gtest_discover_tests(test_game)
endif()

# =============================================================================
# Optional: Thread sanitizer build
# =============================================================================

option(ENABLE_TSAN "Enable thread sanitizer" OFF)
if(ENABLE_TSAN)
    target_compile_options(qbot_exe_debug PRIVATE -fsanitize=thread)
    target_link_options(qbot_exe_debug PRIVATE -fsanitize=thread)
    target_compile_options(qbot_lib_debug PRIVATE -fsanitize=thread)
endif()
