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
# Optional: LibTorch for neural network inference
# =============================================================================

option(ENABLE_INFERENCE "Enable neural network inference with LibTorch" OFF)

if(ENABLE_INFERENCE)
    # Try to use conda environment if available
    if(DEFINED ENV{CONDA_PREFIX})
        set(LIBTORCH_PREFIX $ENV{CONDA_PREFIX})
        message(STATUS "Using conda environment: ${LIBTORCH_PREFIX}")

        # Set include directories
        set(TORCH_INCLUDE_DIRS
            ${LIBTORCH_PREFIX}/include
            ${LIBTORCH_PREFIX}/include/torch/csrc/api/include)

        # Set library directory
        set(TORCH_LIBRARY_DIR ${LIBTORCH_PREFIX}/lib)

        # Manually specify torch libraries (matches old Makefile approach)
        # Use -Wl,--no-as-needed to ensure CUDA libs are linked even if not directly referenced
        set(TORCH_LIBRARIES
            -Wl,--no-as-needed
            ${TORCH_LIBRARY_DIR}/libtorch.so
            ${TORCH_LIBRARY_DIR}/libtorch_cpu.so
            ${TORCH_LIBRARY_DIR}/libtorch_cuda.so
            ${TORCH_LIBRARY_DIR}/libc10.so
            ${TORCH_LIBRARY_DIR}/libc10_cuda.so)

        message(STATUS "LibTorch include dirs: ${TORCH_INCLUDE_DIRS}")
        message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")
    else()
        # Fallback to find_package
        find_package(Torch REQUIRED)
        set(TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
        message(STATUS "LibTorch found via find_package: ${TORCH_LIBRARIES}")
    endif()
endif()

# =============================================================================
# Core Library (compiled separately for each build type)
# =============================================================================

set(QBOT_LIB_SOURCES
    src/tree/StateNode.cpp
    src/util/storage.cpp
    src/util/gui_client.cpp
    src/core/Game.cpp
    src/util/pathfinding.cpp
    src/search/mcts.cpp
)

if(ENABLE_INFERENCE)
    list(APPEND QBOT_LIB_SOURCES
        src/inference/inference.cpp
        src/inference/inference_server.cpp)
endif()

set(QBOT_MAIN_SOURCES
    src/core/QuoridorMain.cpp
)

# =============================================================================
# Debug Build: "qbot" target (default)
# Compiles with debug symbols (-g -O0)
# =============================================================================

add_library(qbot_lib_debug STATIC ${QBOT_LIB_SOURCES})
target_include_directories(qbot_lib_debug PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/websocketpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/json/single_include)
target_compile_options(qbot_lib_debug PRIVATE -g -O0)
# Use standalone ASIO for websocketpp
target_compile_definitions(qbot_lib_debug PUBLIC ASIO_STANDALONE)
# websocketpp uses syntax deprecated in C++20, compile gui_client with C++17
set_source_files_properties(src/util/gui_client.cpp PROPERTIES COMPILE_FLAGS "-std=c++17")
target_link_libraries(qbot_lib_debug PUBLIC pthread)
if(ENABLE_INFERENCE)
    target_include_directories(qbot_lib_debug PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(qbot_lib_debug PUBLIC ${TORCH_LIBRARIES})
    target_compile_definitions(qbot_lib_debug PUBLIC QBOT_ENABLE_INFERENCE)
endif()

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
target_include_directories(qbot_lib_fast PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/websocketpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/json/single_include)
target_compile_options(qbot_lib_fast PRIVATE -O3 -DNDEBUG -march=native -flto)
# Use standalone ASIO for websocketpp
target_compile_definitions(qbot_lib_fast PUBLIC ASIO_STANDALONE)
# websocketpp uses syntax deprecated in C++20, compile gui_client with C++17
set_source_files_properties(src/util/gui_client.cpp PROPERTIES COMPILE_FLAGS "-std=c++17")
target_link_libraries(qbot_lib_fast PUBLIC pthread)
if(ENABLE_INFERENCE)
    target_include_directories(qbot_lib_fast PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(qbot_lib_fast PUBLIC ${TORCH_LIBRARIES})
    target_compile_definitions(qbot_lib_fast PUBLIC QBOT_ENABLE_INFERENCE)
endif()

# =============================================================================
# Leopard: Tree file dumper for training
# =============================================================================

add_executable(leopard_exe src/util/leopard.cpp)
target_link_libraries(leopard_exe PRIVATE qbot_lib_fast)
target_compile_options(leopard_exe PRIVATE -O3 -DNDEBUG -march=native)
set_target_properties(leopard_exe PROPERTIES
    OUTPUT_NAME leopard
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_custom_target(leopard DEPENDS leopard_exe)

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

    # Optimized test builds (default)
    add_executable(test_storage tests/test_storage.cpp)
    target_link_libraries(test_storage PRIVATE qbot_lib_fast GTest::gtest_main)
    target_compile_options(test_storage PRIVATE -O3 -DNDEBUG -march=native -flto)
    target_link_options(test_storage PRIVATE -flto)

    add_executable(test_game tests/test_game.cpp)
    target_link_libraries(test_game PRIVATE qbot_lib_fast GTest::gtest_main)
    target_compile_options(test_game PRIVATE -O3 -DNDEBUG -march=native -flto)
    target_link_options(test_game PRIVATE -flto)

    if(ENABLE_INFERENCE)
        # Regular inference tests - debug build for ctest
        add_executable(test_inference tests/test_inference.cpp)
        target_link_libraries(test_inference PRIVATE qbot_lib_debug GTest::gtest_main)
        target_compile_options(test_inference PRIVATE -g -O0)

        # Inference benchmarks - optimized build, NOT registered with ctest
        add_executable(bench_inference tests/bench_inference.cpp)
        target_link_libraries(bench_inference PRIVATE qbot_lib_fast GTest::gtest_main)
        target_compile_options(bench_inference PRIVATE -O3 -DNDEBUG -march=native -flto)
        target_link_options(bench_inference PRIVATE -flto)
        set_target_properties(bench_inference PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()

    # Debug test builds
    add_executable(test_storage_debug tests/test_storage.cpp)
    target_link_libraries(test_storage_debug PRIVATE qbot_lib_debug GTest::gtest_main)
    target_compile_options(test_storage_debug PRIVATE -g -O0)

    add_executable(test_game_debug tests/test_game.cpp)
    target_link_libraries(test_game_debug PRIVATE qbot_lib_debug GTest::gtest_main)
    target_compile_options(test_game_debug PRIVATE -g -O0)

    if(ENABLE_INFERENCE)
        add_executable(test_inference_debug tests/test_inference.cpp)
        target_link_libraries(test_inference_debug PRIVATE qbot_lib_debug GTest::gtest_main)
        target_compile_options(test_inference_debug PRIVATE -g -O0)
    endif()

    include(GoogleTest)
    gtest_discover_tests(test_storage)
    gtest_discover_tests(test_game)
    if(ENABLE_INFERENCE)
        gtest_discover_tests(test_inference)
    endif()

    # "tests" target - builds all test binaries (excludes benchmarks)
    set(TEST_TARGETS test_storage test_game)
    if(ENABLE_INFERENCE)
        list(APPEND TEST_TARGETS test_inference)
    endif()

    add_custom_target(tests DEPENDS ${TEST_TARGETS})

    # "benchmarks" target - builds benchmark binaries (optimized, not run by ctest)
    if(ENABLE_INFERENCE)
        add_custom_target(benchmarks DEPENDS bench_inference
            COMMENT "Building optimized benchmark binaries (run directly, not via ctest)"
        )
    endif()
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
