// bfloat16 NumPy dtype (issue #3) — now expressed through the reusable NEP-42
// harness (universal_dtype.hpp, issue #7). All the registration machinery lives
// in the harness; this file supplies only the ~handful of type facts for
// sw::universal::bfloat16. Behavior is unchanged and still validated bit-for-bit
// against ml_dtypes.bfloat16.

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL universal_dtypes_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL universal_dtypes_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

#include <cmath>
#include <cstdint>

#include <universal/number/bfloat16/bfloat16.hpp>

#include "universal_dtype.hpp"

namespace {

struct Bfloat16Traits {
    using cpp_t = sw::universal::bfloat16;
    using storage_t = uint16_t;

    static constexpr const char* name = "bfloat16";
    static constexpr const char* scalar_tp_name = "universal_dtypes.bfloat16";
    static constexpr const char* dtype_tp_name = "universal_dtypes.Bfloat16DType";
    static constexpr const char* dtype_attr = "Bfloat16DType";
    static constexpr const char* doc = "bfloat16 scalar (1 sign / 8 exponent / 7 mantissa)";

    static storage_t to_bits(const cpp_t& v) { return v.bits(); }
    static cpp_t from_bits(storage_t b) { cpp_t v; v.setbits(b); return v; }
    static cpp_t from_double(double d) { return cpp_t(static_cast<float>(d)); }
    static double to_double(const cpp_t& v) { return static_cast<double>(v); }
    static bool is_nan(const cpp_t& v) { return std::isnan(static_cast<float>(v)); }
};

}  // namespace

void register_bfloat16(nb::module_& m) {
    register_universal_dtype<Bfloat16Traits>(m);
}
