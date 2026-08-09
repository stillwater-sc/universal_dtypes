// fixpnt<nbits,rbits,arithmetic,bt> NumPy dtype family (issue #29) — Universal's
// fixed-point, bound to NumPy dtypes via the reusable NEP-42 harness. Fixed-point
// gives uniform absolute resolution across its range and exact addition within
// range (no exponent, no rounding on add) — the reason it's a staple of DSP.
//
// Shipped configs use SATURATING arithmetic (out-of-range clamps to maxpos/maxneg
// rather than wrapping, which is the DSP-safe choice):
//   fixpnt16 = fixpnt<16, 8, Saturate, uint16>   (Q8.8: +-128, resolution 2^-8)
//   fixpnt8  = fixpnt<8,  4, Saturate, uint8>     (Q4.4: +-8,   resolution 2^-4)
//
// Fixed-point has no NaN/Inf, so isnan/isinf are always false and isfinite true.

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL universal_dtypes_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL universal_dtypes_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

#include <cstdint>

#include <universal/number/fixpnt/fixpnt.hpp>

#include "universal_dtype.hpp"

namespace {

template <unsigned NBITS, unsigned RBITS, typename Storage>
struct FixpntTraitsBase {
    using cpp_t = sw::universal::fixpnt<NBITS, RBITS, sw::universal::Saturate, Storage>;
    using storage_t = Storage;

    static storage_t to_bits(const cpp_t& v) { return static_cast<storage_t>(v.bits().to_ull()); }
    static cpp_t from_bits(storage_t b) {
        cpp_t v;
        v.setbits(static_cast<uint64_t>(b));
        return v;
    }
    static cpp_t from_double(double d) { return cpp_t(d); }
    static double to_double(const cpp_t& v) { return static_cast<double>(v); }
    static bool is_nan(const cpp_t&) { return false; }  // fixed-point: no NaN
    static bool is_inf(const cpp_t&) { return false; }  // fixed-point: no infinity
    // fixpnt is lossless to double at these sizes, so value ops go through it.
    static bool lt(const cpp_t& a, const cpp_t& b) { return to_double(a) < to_double(b); }
    static bool eq(const cpp_t& a, const cpp_t& b) { return to_double(a) == to_double(b); }
    static bool is_zero(const cpp_t& v) { return to_double(v) == 0.0; }
};

#define UD_FIXPNT_LIST(X)                       \
    X(16, 8, uint16_t, "fixpnt16", Fixpnt16)    \
    X(8, 4, uint8_t, "fixpnt8", Fixpnt8)

#define UD_FIXPNT_DEFINE(NBITS, RBITS, STORE, SNAME, CBASE)                              \
    struct CBASE##Traits : FixpntTraitsBase<NBITS, RBITS, STORE> {                       \
        static constexpr const char* name = SNAME;                                       \
        static constexpr const char* scalar_tp_name = "universal_dtypes." SNAME;         \
        static constexpr const char* dtype_tp_name = "universal_dtypes." #CBASE "DType";  \
        static constexpr const char* dtype_attr = #CBASE "DType";                        \
        static constexpr const char* doc =                                              \
            "fixpnt<" #NBITS "," #RBITS "> saturating fixed-point (uniform resolution)"; \
    };
UD_FIXPNT_LIST(UD_FIXPNT_DEFINE)
#undef UD_FIXPNT_DEFINE

}  // namespace

void register_fixpnts(nb::module_& m) {
#define UD_FIXPNT_REGISTER(NBITS, RBITS, STORE, SNAME, CBASE) \
    register_universal_dtype<CBASE##Traits>(m);
    UD_FIXPNT_LIST(UD_FIXPNT_REGISTER)
#undef UD_FIXPNT_REGISTER
}

#undef UD_FIXPNT_LIST
