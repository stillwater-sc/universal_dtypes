// cfloat<nbits,es,bt,...> NumPy dtype family (issue #8) — Universal's
// configurable float (IEEE-754-style: sign / exponent / fraction, with optional
// subnormals / supernormals / saturation), bound to NumPy dtypes via the reusable
// NEP-42 harness (universal_dtype.hpp).
//
// Shipped configs, both chosen for exact ml_dtypes parity:
//   fp16     = cfloat<16,5,uint16,true,false,false>  == IEEE half (numpy.float16)
//   fp8e5m2  = cfloat<8, 5,uint8, true,false,false>  == ml_dtypes.float8_e5m2
//
// e4m3: NOT shipped here. No Universal type is bit-exact with
// ml_dtypes.float8_e4m3fn (OCP e4m3fn: max 448, no inf, overflow -> NaN);
// cfloat<8,4> is IEEE-style (has inf, max 240) and microfloat e4m3 saturates
// large overflow to 448 rather than NaN. Tracked as a follow-up (see #8 thread).
//
// bfloat16 is the cfloat<16,8,...> config but keeps its dedicated standalone
// implementation (ml_dtypes parity) in bfloat16.cpp — see docs/design.md.

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL universal_dtypes_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL universal_dtypes_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

#include <cstdint>

#include <universal/number/cfloat/cfloat.hpp>

#include "universal_dtype.hpp"

namespace {

// Common cfloat facts; each concrete config only adds its names (via UD_CFLOAT).
// Raw bits come from block(0) — a single storage block for nbits <= 16.
template <unsigned NBITS, unsigned ES, typename Storage>
struct CfloatTraitsBase {
    using cpp_t = sw::universal::cfloat<NBITS, ES, Storage, true, false, false>;
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
    static bool is_inf(const cpp_t& v) { return v.isinf(); }
};

#define UD_CFLOAT_LIST(X)                       \
    X(16, 5, uint16_t, "fp16", Fp16)            \
    X(8, 5, uint8_t, "fp8e5m2", Fp8e5m2)

#define UD_CFLOAT_DEFINE(NBITS, ES, STORE, SNAME, CBASE)                                 \
    struct CBASE##Traits : CfloatTraitsBase<NBITS, ES, STORE> {                          \
        static constexpr const char* name = SNAME;                                       \
        static constexpr const char* scalar_tp_name = "universal_dtypes." SNAME;         \
        static constexpr const char* dtype_tp_name = "universal_dtypes." #CBASE "DType";  \
        static constexpr const char* dtype_attr = #CBASE "DType";                        \
        static constexpr const char* doc =                                              \
            "cfloat<" #NBITS "," #ES "> scalar (IEEE-style; subnormals, inf, NaN)";      \
    };
UD_CFLOAT_LIST(UD_CFLOAT_DEFINE)
#undef UD_CFLOAT_DEFINE

}  // namespace

void register_cfloats(nb::module_& m) {
#define UD_CFLOAT_REGISTER(NBITS, ES, STORE, SNAME, CBASE) register_universal_dtype<CBASE##Traits>(m);
    UD_CFLOAT_LIST(UD_CFLOAT_REGISTER)
#undef UD_CFLOAT_REGISTER
}

#undef UD_CFLOAT_LIST
