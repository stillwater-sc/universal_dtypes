# Find Python — nanobind requires the target name "Python" (not "Python3")
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# nanobind: try find_package first, then FetchContent
find_package(nanobind QUIET)
if(NOT nanobind_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        v2.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(nanobind)
endif()

nanobind_add_module(_core src/universal_dtypes_module.cpp)

# Header-only Universal number systems.
target_link_libraries(_core PRIVATE universal)

# Inject the package version from pyproject.toml (scikit-build-core sets
# SKBUILD_PROJECT_VERSION). A bare `cmake -S . -B build` has no package metadata,
# so it falls back to the "0.0.0-dev" sentinel.
if(DEFINED SKBUILD_PROJECT_VERSION AND NOT "${SKBUILD_PROJECT_VERSION}" STREQUAL "")
    set(_UD_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
    set(_UD_VERSION "0.0.0-dev")
endif()
target_compile_definitions(_core PRIVATE UNIVERSAL_DTYPES_VERSION="${_UD_VERSION}")

install(TARGETS _core DESTINATION universal_dtypes)
