// dd_cascade NumPy dtype (issue #4) — Universal's double-double: an unevaluated
// sum of two IEEE float64 giving ~106 bits of significand (~31 decimal digits).
// Bound through the reusable NEP-42 harness (universal_dtype.hpp).
//
// This is the pattern-setter for MULTI-WORD storage: unlike the small scalar
// types, an element is 16 bytes (two doubles). The harness is storage-agnostic —
// it moves `storage_t` with memcpy and converts via to_bits/from_bits — so a
// two-limb POD `dd_storage {hi, lo}` slots straight in (itemsize 16, align 8).
// td_cascade (#5) and qd_cascade (#6) reuse this directly.
//
// Precision-aware wiring:
//   - to_double() = hi + lo is LOSSY, so dd -> float/double casts are marked
//     UNSAFE (to_float_casting), and comparisons/sort use dd's own operators
//     (lt/eq/is_zero) rather than to_double — full 106-bit ordering.
//   - float64 -> dd is exact (lo = 0).
// Special values follow the underlying doubles (inf/nan propagate).

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL universal_dtypes_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL universal_dtypes_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

#include <universal/number/dd/dd.hpp>

#include "universal_dtype.hpp"

namespace {

// Two-limb element storage: the unevaluated (hi, lo) pair, 16 bytes, trivially
// copyable — exactly what NumPy needs for a fixed-width element.
struct dd_storage {
    double hi;
    double lo;
};

struct DdCascadeTraits {
    using cpp_t = sw::universal::dd;
    using storage_t = dd_storage;

    static constexpr const char* name = "dd_cascade";
    static constexpr const char* scalar_tp_name = "universal_dtypes.dd_cascade";
    static constexpr const char* dtype_tp_name = "universal_dtypes.DdCascadeDType";
    static constexpr const char* dtype_attr = "DdCascadeDType";
    static constexpr const char* doc =
        "double-double scalar (~106-bit significand: unevaluated sum of two float64)";

    // dd -> float64 loses the low limb, so the out-cast is unsafe/lossy.
    static constexpr NPY_CASTING to_float_casting = NPY_UNSAFE_CASTING;

    static storage_t to_bits(const cpp_t& v) { return storage_t{v.high(), v.low()}; }
    static cpp_t from_bits(storage_t b) { return cpp_t(b.hi, b.lo); }
    static cpp_t from_double(double d) { return cpp_t(d); }  // exact: lo = 0
    static double to_double(const cpp_t& v) { return static_cast<double>(v); }  // hi + lo (lossy)
    static bool is_nan(const cpp_t& v) { return v.isnan(); }
    static bool is_inf(const cpp_t& v) { return v.isinf(); }
    // Full 106-bit comparisons via dd's own operators (to_double would collapse
    // values that differ below float64 precision).
    static bool lt(const cpp_t& a, const cpp_t& b) { return a < b; }
    static bool eq(const cpp_t& a, const cpp_t& b) { return a == b; }
    static bool is_zero(const cpp_t& v) { return v.iszero(); }
};

}  // namespace

void register_dd_cascade(nb::module_& m) {
    register_universal_dtype<DdCascadeTraits>(m);
}
