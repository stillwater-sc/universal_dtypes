// td_cascade NumPy dtype (issue #5) — Universal's triple-double: an unevaluated
// sum of three IEEE float64 giving ~159 bits of significand (~47 decimal digits).
// A direct extension of dd_cascade's multi-word pattern (see dd.cpp): the only
// differences are three limbs instead of two (24-byte element) and the concrete
// Universal type.
//
// Casting/comparison rules mirror dd_cascade: float64 -> td exact, td -> float64
// unsafe/lossy, comparisons/sort run at full precision via the type's operators.

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL universal_dtypes_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL universal_dtypes_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

#include <universal/number/td_cascade/td_cascade.hpp>

#include "universal_dtype.hpp"

namespace {

// Three-limb element storage: the unevaluated (l0, l1, l2) triple, 24 bytes.
struct td_storage {
    double l0;
    double l1;
    double l2;
};

struct TdCascadeTraits {
    using cpp_t = sw::universal::td_cascade;
    using storage_t = td_storage;

    static constexpr const char* name = "td_cascade";
    static constexpr const char* scalar_tp_name = "universal_dtypes.td_cascade";
    static constexpr const char* dtype_tp_name = "universal_dtypes.TdCascadeDType";
    static constexpr const char* dtype_attr = "TdCascadeDType";
    static constexpr const char* doc =
        "triple-double scalar (~159-bit significand: unevaluated sum of three float64)";

    // td -> float64 drops the two low limbs, so the out-cast is unsafe/lossy.
    static constexpr NPY_CASTING to_float_casting = NPY_UNSAFE_CASTING;

    static storage_t to_bits(const cpp_t& v) { return storage_t{v[0], v[1], v[2]}; }
    static cpp_t from_bits(storage_t b) { return cpp_t(b.l0, b.l1, b.l2); }
    static cpp_t from_double(double d) { return cpp_t(d); }  // exact: low limbs = 0
    static double to_double(const cpp_t& v) { return static_cast<double>(v); }  // lossy
    static bool is_nan(const cpp_t& v) { return v.isnan(); }
    static bool is_inf(const cpp_t& v) { return v.isinf(); }
    // Full ~159-bit comparisons via td_cascade's own operators.
    static bool lt(const cpp_t& a, const cpp_t& b) { return a < b; }
    static bool eq(const cpp_t& a, const cpp_t& b) { return a == b; }
    static bool is_zero(const cpp_t& v) { return v.iszero(); }
};

}  // namespace

void register_td_cascade(nb::module_& m) {
    register_universal_dtype<TdCascadeTraits>(m);
}
