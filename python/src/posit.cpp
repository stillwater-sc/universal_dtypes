// posit<nbits,es> NumPy dtype family (issues #7, #16) — the first templated
// Universal type bound to NumPy dtypes, via the reusable NEP-42 harness
// (universal_dtype.hpp).
//
// Adding a config is one line in UD_POSIT_LIST below. `nbits` need not be a power
// of two and need not equal the storage width: each element is stored in the
// smallest containing unsigned integer (storage_for<nbits>), with the unused
// high bits zero. Casts/ufuncs/pickle all go through the posit type, not raw
// bytes, so the padding is invisible. (numpy is byte-granular, so e.g. posit12
// occupies 2 bytes; true bit-packing is outside numpy's dtype model.)
//
// Naming: `posit{nbits}` is the es=2 standard; other exponent sizes are
// `posit{nbits}e{es}` (e.g. posit8e0). Posit specifics: NaR maps onto isnan,
// non-finite float -> NaR, float<->posit rounds to nearest posit.

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL universal_dtypes_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL universal_dtypes_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

#include <cstdint>
#include <type_traits>

#include <universal/number/posit/posit.hpp>

#include "universal_dtype.hpp"

namespace {

// Smallest unsigned integer that holds `nbits` bits — the numpy element storage.
template <unsigned N>
using posit_storage_t = std::conditional_t<
    (N <= 8), uint8_t,
    std::conditional_t<(N <= 16), uint16_t,
                       std::conditional_t<(N <= 32), uint32_t, uint64_t>>>;

// Common posit facts; each concrete config only adds its names (via UD_POSIT).
template <unsigned NBITS, unsigned ES>
struct PositTraitsBase {
    static_assert(ES < NBITS, "posit requires es < nbits");
    using cpp_t = sw::universal::posit<NBITS, ES>;
    using storage_t = posit_storage_t<NBITS>;

    // Read raw bits via bits().to_ull(); posit::encoding() is unusable on
    // Universal main (it calls the nonexistent blockbinary::to_ullong()).
    static storage_t to_bits(const cpp_t& v) { return static_cast<storage_t>(v.bits().to_ull()); }
    static cpp_t from_bits(storage_t b) {
        cpp_t v;
        v.setbits(static_cast<uint64_t>(b));
        return v;
    }
    static cpp_t from_double(double d) { return cpp_t(d); }
    static double to_double(const cpp_t& v) { return static_cast<double>(v); }
    static bool is_nan(const cpp_t& v) { return v.isnar(); }
    static bool is_inf(const cpp_t&) { return false; }  // posit has NaR, no infinity
    // posit is lossless to double, so value comparisons go through it.
    static bool lt(const cpp_t& a, const cpp_t& b) { return to_double(a) < to_double(b); }
    static bool eq(const cpp_t& a, const cpp_t& b) { return to_double(a) == to_double(b); }
    static bool is_zero(const cpp_t& v) { return to_double(v) == 0.0; }
};

// The shipped set. One line per config; `es < nbits` is enforced at compile time.
// Standard es=2 sizes keep the bare `posit{nbits}` name; other configs use the
// `posit{nbits}e{es}` form. Adding a config here is the whole change.
#define UD_POSIT_LIST(X)          \
    X(8, 2, "posit8", Posit8)     \
    X(16, 2, "posit16", Posit16)  \
    X(32, 2, "posit32", Posit32)  \
    X(64, 2, "posit64", Posit64)  \
    X(8, 0, "posit8e0", Posit8e0) \
    X(8, 1, "posit8e1", Posit8e1) \
    X(16, 1, "posit16e1", Posit16e1) \
    X(12, 2, "posit12", Posit12)  \
    X(20, 2, "posit20", Posit20)  \
    X(24, 2, "posit24", Posit24)  \
    X(28, 2, "posit28", Posit28)  \
    X(40, 2, "posit40", Posit40)  \
    X(48, 2, "posit48", Posit48)

#define UD_POSIT_DEFINE(NBITS, ES, SNAME, CBASE)                                        \
    struct CBASE##Traits : PositTraitsBase<NBITS, ES> {                                 \
        static constexpr const char* name = SNAME;                                      \
        static constexpr const char* scalar_tp_name = "universal_dtypes." SNAME;        \
        static constexpr const char* dtype_tp_name = "universal_dtypes." #CBASE "DType"; \
        static constexpr const char* dtype_attr = #CBASE "DType";                       \
        static constexpr const char* doc =                                              \
            "posit<" #NBITS "," #ES "> scalar (tapered precision, NaR)";                \
    };
UD_POSIT_LIST(UD_POSIT_DEFINE)
#undef UD_POSIT_DEFINE

}  // namespace

void register_posits(nb::module_& m) {
#define UD_POSIT_REGISTER(NBITS, ES, SNAME, CBASE) register_universal_dtype<CBASE##Traits>(m);
    UD_POSIT_LIST(UD_POSIT_REGISTER)
#undef UD_POSIT_REGISTER
}

#undef UD_POSIT_LIST
