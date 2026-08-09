// lns<nbits,rbits,bt> NumPy dtype family (issue #9) — Universal's logarithmic
// number system bound to NumPy dtypes via the reusable NEP-42 harness
// (universal_dtype.hpp), following the posit/cfloat codegen-table pattern.
//
// LNS stores a sign and the base-2 logarithm of the magnitude, so multiply/divide
// become add/subtract of the stored logs (cheap, exact in the exponent), while
// add/subtract are the hard operations (Gaussian-log). All of that lives in
// Universal's lns operators — the harness just moves bits and calls them.
//
// Shipped configs (Universal's canonical splits, used across its own tooling):
//   lns16 = lns<16, 8, uint16>
//   lns32 = lns<32,16, uint32>
//
// Special values: LNS has a dedicated encoding for zero and for NaN, and NO
// infinity (isinf() is always false) — so is_inf is false and isfinite = !isnan.
// There is no ml_dtypes counterpart; conversion/arithmetic are validated against
// Universal's own lns.

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL universal_dtypes_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL universal_dtypes_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

#include <cstdint>

#include <universal/number/lns/lns.hpp>

#include "universal_dtype.hpp"

namespace {

// Common lns facts; each concrete config only adds its names (via UD_LNS).
// Raw bits come from block(0) — a single storage block for nbits <= 32 with the
// matching block type.
template <unsigned NBITS, unsigned RBITS, typename Storage>
struct LnsTraitsBase {
    using cpp_t = sw::universal::lns<NBITS, RBITS, Storage>;
    using storage_t = Storage;

    static storage_t to_bits(const cpp_t& v) { return static_cast<storage_t>(v.block(0)); }
    static cpp_t from_bits(storage_t b) {
        cpp_t v;
        v.setbits(static_cast<uint64_t>(b));
        return v;
    }
    static cpp_t from_double(double d) { return cpp_t(d); }
    static double to_double(const cpp_t& v) { return static_cast<double>(v); }
    static bool is_nan(const cpp_t& v) { return v.isnan(); }
    static bool is_inf(const cpp_t&) { return false; }  // LNS has zero/NaN encodings, no infinity
};

#define UD_LNS_LIST(X)                    \
    X(16, 8, uint16_t, "lns16", Lns16)    \
    X(32, 16, uint32_t, "lns32", Lns32)

#define UD_LNS_DEFINE(NBITS, RBITS, STORE, SNAME, CBASE)                                 \
    struct CBASE##Traits : LnsTraitsBase<NBITS, RBITS, STORE> {                          \
        static constexpr const char* name = SNAME;                                       \
        static constexpr const char* scalar_tp_name = "universal_dtypes." SNAME;         \
        static constexpr const char* dtype_tp_name = "universal_dtypes." #CBASE "DType";  \
        static constexpr const char* dtype_attr = #CBASE "DType";                        \
        static constexpr const char* doc =                                              \
            "lns<" #NBITS "," #RBITS "> scalar (logarithmic number system; zero/NaN)";   \
    };
UD_LNS_LIST(UD_LNS_DEFINE)
#undef UD_LNS_DEFINE

}  // namespace

void register_lns(nb::module_& m) {
#define UD_LNS_REGISTER(NBITS, RBITS, STORE, SNAME, CBASE) register_universal_dtype<CBASE##Traits>(m);
    UD_LNS_LIST(UD_LNS_REGISTER)
#undef UD_LNS_REGISTER
}

#undef UD_LNS_LIST
