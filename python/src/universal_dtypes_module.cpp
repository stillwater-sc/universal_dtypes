// universal_dtypes core extension.
//
// This translation unit owns the NumPy C-API import (the shared
// PY_ARRAY_UNIQUE_SYMBOL / PY_UFUNC_UNIQUE_SYMBOL tables); the dtype
// translation units (bfloat16.cpp, posit.cpp) reference the same tables via
// NO_IMPORT_ARRAY / NO_IMPORT_UFUNC and register through the NEP-42 harness in
// universal_dtype.hpp.

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL universal_dtypes_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL universal_dtypes_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <universal/number/posit/posit.hpp>

#include <stdexcept>

namespace nb = nanobind;

#ifndef UNIVERSAL_DTYPES_VERSION
#define UNIVERSAL_DTYPES_VERSION "0.0.0-dev"
#endif

// dtype registrars (own translation units, sharing the numpy API tables above).
void register_bfloat16(nb::module_& m);  // python/src/bfloat16.cpp
void register_posits(nb::module_& m);    // python/src/posit.cpp
void register_cfloats(nb::module_& m);   // python/src/cfloat.cpp
void register_lns(nb::module_& m);       // python/src/lns.cpp
void register_dd_cascade(nb::module_& m); // python/src/dd.cpp

NB_MODULE(_core, m) {
    if (_import_array() < 0) throw std::runtime_error("numpy multiarray import failed");
    if (_import_umath() < 0) throw std::runtime_error("numpy umath import failed");

    register_bfloat16(m);
    register_posits(m);
    register_cfloats(m);
    register_lns(m);
    register_dd_cascade(m);

    m.doc() = "universal_dtypes core — NumPy dtypes for the Universal number systems";
    m.attr("__version__") = UNIVERSAL_DTYPES_VERSION;

    // Proof-of-life: round a Python float through a posit<16,2> and back. This
    // exercises the Universal dependency directly (independent of the dtype API).
    m.def(
        "posit16_roundtrip",
        [](double x) {
            sw::universal::posit<16, 2> p{x};
            return double(p);
        },
        nb::arg("x"),
        "Round a Python float through a posit<16,2> and back to double.");

    m.def(
        "build_info",
        []() {
            nb::dict d;
            d["version"] = std::string(UNIVERSAL_DTYPES_VERSION);
            d["universal"] = true;  // Universal headers linked and compiled in
            d["dtypes"] = true;     // NumPy dtype registration is built (bfloat16 + posit family)
            return d;
        },
        "Compile-time facts about this build.");
}
