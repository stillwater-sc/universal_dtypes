cmake_minimum_required(VERSION 3.22)
project(universal_dtypes CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC: report the correct __cplusplus value so C++20 headers like <bit> are
# detected by Universal's bit_cast.hpp.
if(MSVC)
    add_compile_options(/Zc:__cplusplus)
endif()

# --- Universal number library (header-only) ---
# Provides posit<nbits,es>, cfloat<nbits,es,...>, fixpnt<nbits,rbits>,
# lns<nbits,rbits>, etc. All of Universal's own build targets are turned off so
# only the headers are consumed.
find_package(universal QUIET)
if(NOT universal_FOUND)
    include(FetchContent)
    set(BUILD_ALL_LIBS OFF CACHE BOOL "" FORCE)
    set(BUILD_DEMONSTRATION OFF CACHE BOOL "" FORCE)
    set(BUILD_NUMBERS OFF CACHE BOOL "" FORCE)
    set(BUILD_NUMERICS OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_REGRESSION_SUITES OFF CACHE BOOL "" FORCE)
    set(BUILD_CMD_LINE_TOOLS OFF CACHE BOOL "" FORCE)
    set(BUILD_EDUCATION OFF CACHE BOOL "" FORCE)
    set(BUILD_APPLICATIONS OFF CACHE BOOL "" FORCE)
    set(BUILD_PLAYGROUND OFF CACHE BOOL "" FORCE)
    set(BUILD_VALIDATION_HW OFF CACHE BOOL "" FORCE)
    set(BUILD_C_API_LIB OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        universal
        GIT_REPOSITORY https://github.com/stillwater-sc/universal.git
        GIT_TAG        main
        GIT_SHALLOW    TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(universal)
endif()

add_subdirectory(python)
