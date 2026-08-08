// universal_dtypes core extension.
//
// SCAFFOLD: this module exists to stand up the packaging + release pipeline and
// to prove the Universal build wiring end to end. The actual NumPy custom-dtype
// registration (posit/cfloat/fixpnt/lns) is issue #14 and has NOT been built
// yet — see docs/design.md. Everything below is a placeholder.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <universal/number/posit/posit.hpp>

namespace nb = nanobind;

#ifndef UNIVERSAL_DTYPES_VERSION
#define UNIVERSAL_DTYPES_VERSION "0.0.0-dev"
#endif

NB_MODULE(_core, m) {
    m.doc() = "universal_dtypes core — NumPy dtypes for Universal number systems (scaffold)";
    m.attr("__version__") = UNIVERSAL_DTYPES_VERSION;

    // Proof-of-life: round a Python float through a posit<16,2> and back. This
    // exercises the Universal dependency (FetchContent + headers + compile); it
    // is NOT the dtype API, which registers posit16 et al. with NumPy directly.
    m.def(
        "posit16_roundtrip",
        [](double x) {
            sw::universal::posit<16, 2> p{x};
            return double(p);
        },
        nb::arg("x"),
        "Round a Python float through a posit<16,2> and back to double. A "
        "placeholder proving the Universal build wiring; not the dtype API.");

    m.def(
        "build_info",
        []() {
            nb::dict d;
            d["version"] = std::string(UNIVERSAL_DTYPES_VERSION);
            d["universal"] = true;   // Universal headers linked and compiled in
            d["dtypes"] = false;     // NumPy dtype registration not yet built
            return d;
        },
        "Compile-time facts about this build.");
}
