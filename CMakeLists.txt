cmake_minimum_required(VERSION 3.20)
project(columnstore LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------
include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)

set(COLUMNSTORE_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
string(TOLOWER "${COLUMNSTORE_ARCH}" COLUMNSTORE_ARCH_LOWER)

set(COLUMNSTORE_X86_64 OFF)
if(COLUMNSTORE_ARCH_LOWER MATCHES "(x86_64|amd64)")
    set(COLUMNSTORE_X86_64 ON)
endif()

# Compile-time AVX2 capability of the toolchain. Runtime detection still
# happens via __builtin_cpu_supports.
set(COLUMNSTORE_HAS_AVX2 OFF)
if(COLUMNSTORE_X86_64)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2_FLAG)
    if(COMPILER_SUPPORTS_AVX2_FLAG)
        set(COLUMNSTORE_HAS_AVX2 ON)
    endif()
endif()

message(STATUS "columnstore: arch=${COLUMNSTORE_ARCH_LOWER} x86_64=${COLUMNSTORE_X86_64} avx2=${COLUMNSTORE_HAS_AVX2}")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(COLUMNSTORE_BUILD_TESTS "Build unit and integration tests" ON)
option(COLUMNSTORE_BUILD_BENCH "Build benchmarks" ON)
option(COLUMNSTORE_BUILD_FUZZ "Build libFuzzer harnesses (clang only)" OFF)
option(COLUMNSTORE_ENABLE_ASAN "Enable AddressSanitizer + UBSan" OFF)
option(COLUMNSTORE_FORCE_SCALAR "Force scalar path even on x86_64" OFF)
option(COLUMNSTORE_ENABLE_COVERAGE "Build with gcov instrumentation" OFF)

# ---------------------------------------------------------------------------
# Global flags
# ---------------------------------------------------------------------------
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wno-unused-parameter
)

if(COLUMNSTORE_ENABLE_ASAN)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g)
    add_link_options(-fsanitize=address,undefined)
endif()

if(COLUMNSTORE_FORCE_SCALAR)
    add_compile_definitions(COLUMNSTORE_FORCE_SCALAR=1)
endif()

if(COLUMNSTORE_ENABLE_COVERAGE)
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()

# ---------------------------------------------------------------------------
# GoogleTest via FetchContent
# ---------------------------------------------------------------------------
if(COLUMNSTORE_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# ---------------------------------------------------------------------------
# Library targets
# ---------------------------------------------------------------------------
add_library(columnstore_core STATIC
    src/core/column.cpp
    src/core/batch.cpp
    src/core/schema.cpp
    src/core/dict.cpp
    src/rle/encoder.cpp
    src/rle/decoder.cpp
    src/simd/cpu_detect.cpp
    src/simd/filter_int32_scalar.cpp
    src/simd/sum_int32_scalar.cpp
    src/exec/source.cpp
    src/exec/filter.cpp
    src/exec/aggregate.cpp
    src/exec/count_distinct.cpp
    src/exec/sink.cpp
    src/exec/pipeline.cpp
    src/obs/histogram.cpp
)
target_include_directories(columnstore_core PUBLIC src)

if(COLUMNSTORE_HAS_AVX2)
    add_library(columnstore_avx2 OBJECT
        src/simd/filter_int32_avx2.cpp
        src/simd/sum_int32_avx2.cpp
        src/simd/filter_dict_eq_avx2.cpp
    )
    target_compile_options(columnstore_avx2 PRIVATE -mavx2)
    target_include_directories(columnstore_avx2 PUBLIC src)
    target_compile_definitions(columnstore_avx2 PRIVATE COLUMNSTORE_COMPILED_WITH_AVX2=1)
    target_compile_definitions(columnstore_core PUBLIC COLUMNSTORE_COMPILED_WITH_AVX2=1)
    target_link_libraries(columnstore_core PUBLIC columnstore_avx2)
endif()

add_executable(columnstore src/main.cpp)
target_link_libraries(columnstore PRIVATE columnstore_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(COLUMNSTORE_BUILD_TESTS)
    add_executable(columnstore_unit_tests
        tests/unit/column_test.cpp
        tests/unit/rle_test.cpp
        tests/unit/filter_test.cpp
        tests/unit/aggregate_test.cpp
        tests/unit/pipeline_test.cpp
        tests/unit/property_test.cpp
        tests/unit/dict_test.cpp
    )
    target_link_libraries(columnstore_unit_tests PRIVATE columnstore_core gtest_main gmock)
    add_test(NAME columnstore_unit_tests COMMAND columnstore_unit_tests)

    add_executable(columnstore_integration_tests
        tests/integration/query_test.cpp
    )
    target_link_libraries(columnstore_integration_tests PRIVATE columnstore_core gtest_main)
    add_test(NAME columnstore_integration_tests COMMAND columnstore_integration_tests)
endif()

# ---------------------------------------------------------------------------
# Fuzz harnesses (clang only)
# ---------------------------------------------------------------------------
if(COLUMNSTORE_BUILD_FUZZ)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        message(FATAL_ERROR "Fuzz harnesses require clang for libFuzzer")
    endif()
    add_executable(rle_fuzz tests/fuzz/rle_fuzz.cpp)
    target_link_libraries(rle_fuzz PRIVATE columnstore_core)
    target_compile_options(rle_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
    target_link_options(rle_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)

    add_executable(filter_fuzz tests/fuzz/filter_fuzz.cpp)
    target_link_libraries(filter_fuzz PRIVATE columnstore_core)
    target_compile_options(filter_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
    target_link_options(filter_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(COLUMNSTORE_BUILD_BENCH)
    add_executable(scan_bench bench/scan_bench.cpp)
    target_link_libraries(scan_bench PRIVATE columnstore_core)
    target_include_directories(scan_bench PRIVATE bench)

    add_executable(filter_bench bench/filter_bench.cpp)
    target_link_libraries(filter_bench PRIVATE columnstore_core)
    target_include_directories(filter_bench PRIVATE bench)

    add_executable(aggregate_bench bench/aggregate_bench.cpp)
    target_link_libraries(aggregate_bench PRIVATE columnstore_core)
    target_include_directories(aggregate_bench PRIVATE bench)

    add_executable(padding_bench bench/padding_bench.cpp)
    target_link_libraries(padding_bench PRIVATE columnstore_core)
    target_include_directories(padding_bench PRIVATE bench)

    add_executable(generate_data bench/data/generate.cpp)
    target_link_libraries(generate_data PRIVATE columnstore_core)
endif()
