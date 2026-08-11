// takum<nbits,rbits> NumPy dtype family (issue #63) — Universal's tapered
// logarithmic-range format, bound through the reusable NEP-42 harness
// (universal_dtype.hpp).
//
// This is the LINEAR takum (Hunhold, 2024, arXiv:2404.18603), the only takum
// encoding Universal currently provides. Value = (-1)^sign * (1 + f) * 2^c, with
// the magnitude laid out as [S:1][D:1][R:rbits][C:r bits][M:p bits]. The
// logarithmic variant is tracked upstream as a separate `takum_log` type, so the
// bare `takum{nbits}` names here stay the linear encoding permanently.
//
// What makes takum distinct from posit — the reason both are worth shipping — is
// that its dynamic range is essentially FIXED regardless of width (~5.8e76 from
// 16 bits up), so extra bits buy precision rather than range. posit's range
// instead grows by tens of orders of magnitude per doubling of nbits.
//
// Like posit, takum has a single exceptional value NaR (no infinity, no NaN
// payloads), and storage is two's complement with zero = 0x00..0, NaR = 0x80..0.
//
// KNOWN LIMITATION — takum64 arithmetic is capped at double precision.
// Universal implements takum's + - * / by converting both operands to double,
// computing there, and converting back (see takum_impl.hpp operator+= etc).
// For takum8/16/32 that is exact: their significands are far under double's 53
// bits, so it is a single correct rounding, the same rule this package already
// documents for the math ufuncs. takum64's significand reaches ~59 bits, so its
// operands are rounded to double *before* the operation and detail below that is
// lost. Encoding, casts, comparisons and sort still carry the full width (lt/eq
// below use takum's own operators, not to_double), so only arithmetic is capped.
// posit64 does NOT share this — Universal gives it native arithmetic. Pinned by
// test_takum64_arithmetic_is_limited_to_double.
//
// Adding a config is one line in UD_TAKUM_LIST below.

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL universal_dtypes_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL universal_dtypes_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

#include <cstdint>
#include <type_traits>

#include <universal/number/takum/takum.hpp>

#include "universal_dtype.hpp"

namespace {

// Smallest unsigned integer that holds `nbits` bits — the numpy element storage.
// Using it as takum's block type too keeps the value in a single block, so
// block(0) is the whole encoding (the same trick cfloat.cpp uses).
template <unsigned N>
using takum_storage_t = std::conditional_t<
    (N <= 8), uint8_t,
    std::conditional_t<(N <= 16), uint16_t,
                       std::conditional_t<(N <= 32), uint32_t, uint64_t>>>;

// Common takum facts; each concrete config only adds its names (via UD_TAKUM).
template <unsigned NBITS, unsigned RBITS>
struct TakumTraitsBase {
    using storage_t = takum_storage_t<NBITS>;
    using cpp_t = sw::universal::takum<NBITS, RBITS, storage_t>;

    // The mantissa reaches p = nbits - 2 - rbits bits, so the significand
    // (with the implicit leading 1) reaches nbits - 1 - rbits. Beyond double's
    // 53 the out-cast is lossy and comparisons must not route through double.
    // Measured: takum64's adjacent encodings collapse to the same double and its
    // double round-trip is inexact, while takum8/16/32 round-trip exactly.
    static constexpr bool wider_than_double = (NBITS - 1 - RBITS) > 53;
    static constexpr NPY_CASTING to_float_casting =
        wider_than_double ? NPY_UNSAFE_CASTING : NPY_SAFE_CASTING;

    static storage_t to_bits(const cpp_t& v) { return static_cast<storage_t>(v.block(0)); }
    static cpp_t from_bits(storage_t b) {
        cpp_t v;
        v.setbits(static_cast<uint64_t>(b));
        return v;
    }
    static cpp_t from_double(double d) { return cpp_t(d); }
    static double to_double(const cpp_t& v) { return static_cast<double>(v); }
    static bool is_nan(const cpp_t& v) { return v.isnar(); }
    static bool is_inf(const cpp_t&) { return false; }  // takum has NaR, no infinity
    // Comparisons use takum's own operators rather than to_double, so takum64
    // orders at full precision. (Correct for the narrow configs too, and the
    // cost is the same, so there is no reason to special-case them.)
    static bool lt(const cpp_t& a, const cpp_t& b) { return a < b; }
    static bool eq(const cpp_t& a, const cpp_t& b) { return a == b; }
    static bool is_zero(const cpp_t& v) { return v.iszero(); }
};

// The shipped set. rbits = 3 is the takum spec's regime width and Universal's
// default; the family is curated to the four standard widths for the same
// binary-size reason the posit family is (#16).
#define UD_TAKUM_LIST(X)             \
    X(8, 3, "takum8", Takum8)        \
    X(16, 3, "takum16", Takum16)     \
    X(32, 3, "takum32", Takum32)     \
    X(64, 3, "takum64", Takum64)

#define UD_TAKUM_DEFINE(NBITS, RBITS, SNAME, CBASE)                                      \
    struct CBASE##Traits : TakumTraitsBase<NBITS, RBITS> {                               \
        static constexpr const char* name = SNAME;                                       \
        static constexpr const char* scalar_tp_name = "universal_dtypes." SNAME;         \
        static constexpr const char* dtype_tp_name = "universal_dtypes." #CBASE "DType"; \
        static constexpr const char* dtype_attr = #CBASE "DType";                        \
        static constexpr const char* doc =                                               \
            "linear takum<" #NBITS "," #RBITS "> scalar (tapered precision, "            \
            "width-independent dynamic range, NaR)";                                     \
    };
UD_TAKUM_LIST(UD_TAKUM_DEFINE)
#undef UD_TAKUM_DEFINE

}  // namespace

void register_takums(nb::module_& m) {
#define UD_TAKUM_REGISTER(NBITS, RBITS, SNAME, CBASE) register_universal_dtype<CBASE##Traits>(m);
    UD_TAKUM_LIST(UD_TAKUM_REGISTER)
#undef UD_TAKUM_REGISTER
}

#undef UD_TAKUM_LIST
