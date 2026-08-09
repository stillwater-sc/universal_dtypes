// qd_cascade NumPy dtype (issue #6) — Universal's quad-double: an unevaluated sum
// of four IEEE float64 giving ~212 bits of significand (~62 decimal digits). The
// highest cascade tier and the last of the cascade family; a direct extension of
// dd_cascade/td_cascade's multi-word pattern (four limbs, 32-byte element).
//
// Casting/comparison rules mirror the other cascades: float64 -> qd exact,
// qd -> float64 unsafe/lossy, comparisons/sort at full precision.

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL universal_dtypes_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL universal_dtypes_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

#include <universal/number/qd_cascade/qd_cascade.hpp>

#include "universal_dtype.hpp"

namespace {

// Four-limb element storage: the unevaluated (l0, l1, l2, l3) quadruple, 32 bytes.
struct qd_storage {
    double l0;
    double l1;
    double l2;
    double l3;
};

struct QdCascadeTraits {
    using cpp_t = sw::universal::qd_cascade;
    using storage_t = qd_storage;

    static constexpr const char* name = "qd_cascade";
    static constexpr const char* scalar_tp_name = "universal_dtypes.qd_cascade";
    static constexpr const char* dtype_tp_name = "universal_dtypes.QdCascadeDType";
    static constexpr const char* dtype_attr = "QdCascadeDType";
    static constexpr const char* doc =
        "quad-double scalar (~212-bit significand: unevaluated sum of four float64)";

    // qd -> float64 drops the three low limbs, so the out-cast is unsafe/lossy.
    static constexpr NPY_CASTING to_float_casting = NPY_UNSAFE_CASTING;

    static storage_t to_bits(const cpp_t& v) { return storage_t{v[0], v[1], v[2], v[3]}; }
    static cpp_t from_bits(storage_t b) { return cpp_t(b.l0, b.l1, b.l2, b.l3); }
    static cpp_t from_double(double d) { return cpp_t(d); }  // exact: low limbs = 0
    static double to_double(const cpp_t& v) { return static_cast<double>(v); }  // lossy
    static bool is_nan(const cpp_t& v) { return v.isnan(); }
    static bool is_inf(const cpp_t& v) { return v.isinf(); }
    // Full ~212-bit comparisons via qd_cascade's own operators.
    static bool lt(const cpp_t& a, const cpp_t& b) { return a < b; }
    static bool eq(const cpp_t& a, const cpp_t& b) { return a == b; }
    static bool is_zero(const cpp_t& v) { return v.iszero(); }
};

}  // namespace

void register_qd_cascade(nb::module_& m) {
    register_universal_dtype<QdCascadeTraits>(m);
}
