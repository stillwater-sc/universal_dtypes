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
# lns<nbits,rbits>, takum<nbits,rbits>, etc. All of Universal's own build targets
# are turned off so only the headers are consumed.
#
# PINNED to a release tag, not a branch (issue #66). Universal supplies the
# arithmetic behind every dtype we ship, so tracking `main` made two builds of
# the same universal_dtypes version able to embed different numerics with nothing
# recording which — an upstream change would reach wheels without a commit here.
# Bumping this is a deliberate, reviewable change and should land as its own PR
# (see docs/RELEASING.md).
#
# Single source of truth: the same value drives the FetchContent tag and the
# version reported by universal_dtypes.build_info(), so the two cannot drift.
set(UD_UNIVERSAL_TAG "v4.7.9")

find_package(universal QUIET)
if(universal_FOUND)
    # A pre-installed Universal takes precedence over the pin, so report what was
    # actually compiled against rather than the tag we asked for.
    if(DEFINED universal_VERSION AND NOT "${universal_VERSION}" STREQUAL "")
        set(UD_UNIVERSAL_VERSION "${universal_VERSION} (found package)")
    else()
        set(UD_UNIVERSAL_VERSION "unknown (found package)")
    endif()
    message(STATUS "Universal: using pre-installed package (${UD_UNIVERSAL_VERSION}), not the ${UD_UNIVERSAL_TAG} pin")
else()
    set(UD_UNIVERSAL_VERSION "${UD_UNIVERSAL_TAG}")
    message(STATUS "Universal: fetching pinned ${UD_UNIVERSAL_TAG}")
endif()
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
        GIT_TAG        ${UD_UNIVERSAL_TAG}
        GIT_SHALLOW    TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(universal)
endif()

add_subdirectory(python)
