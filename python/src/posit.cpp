// posit<nbits,2> NumPy dtype family (issue #7) — the first templated Universal
// type bound to NumPy dtypes, via the reusable NEP-42 harness
// (universal_dtype.hpp). Each standard size is one Traits struct + one
// register_universal_dtype<> call; adding a new posit config is a one-liner.
//
// Posit specifics handled here: the single exceptional value NaR ("Not a Real")
// maps onto isnan; float<->posit rounds to nearest posit; non-finite float ->
// NaR. Arithmetic and comparisons come from Universal's posit operators.

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL universal_dtypes_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL universal_dtypes_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

#include <cstdint>

#include <universal/number/posit/posit.hpp>

#include "universal_dtype.hpp"

namespace {

// One traits struct per standard size. `storage_t` is the uintN_t matching the
// posit's bit width (the raw element storage in the numpy buffer); `cpp_t` is the
// Universal posit. encoding()/setbits() move raw bits in and out.
template <unsigned NBITS, typename Storage>
struct PositTraitsBase {
    using cpp_t = sw::universal::posit<NBITS, 2>;
    using storage_t = Storage;

    // Avoid posit::encoding() — it calls blockbinary::to_ullong(), which doesn't
    // exist on Universal main (the block exposes to_ull()). Read the raw block
    // directly instead.
    static storage_t to_bits(const cpp_t& v) { return static_cast<storage_t>(v.bits().to_ull()); }
    static cpp_t from_bits(storage_t b) {
        cpp_t v;
        v.setbits(static_cast<uint64_t>(b));
        return v;
    }
    static cpp_t from_double(double d) { return cpp_t(d); }
    static double to_double(const cpp_t& v) { return static_cast<double>(v); }
    static bool is_nan(const cpp_t& v) { return v.isnar(); }
};

struct Posit8Traits : PositTraitsBase<8, uint8_t> {
    static constexpr const char* name = "posit8";
    static constexpr const char* scalar_tp_name = "universal_dtypes.posit8";
    static constexpr const char* dtype_tp_name = "universal_dtypes.Posit8DType";
    static constexpr const char* dtype_attr = "Posit8DType";
    static constexpr const char* doc = "posit<8,2> scalar (tapered precision, NaR)";
};

struct Posit16Traits : PositTraitsBase<16, uint16_t> {
    static constexpr const char* name = "posit16";
    static constexpr const char* scalar_tp_name = "universal_dtypes.posit16";
    static constexpr const char* dtype_tp_name = "universal_dtypes.Posit16DType";
    static constexpr const char* dtype_attr = "Posit16DType";
    static constexpr const char* doc = "posit<16,2> scalar (tapered precision, NaR)";
};

struct Posit32Traits : PositTraitsBase<32, uint32_t> {
    static constexpr const char* name = "posit32";
    static constexpr const char* scalar_tp_name = "universal_dtypes.posit32";
    static constexpr const char* dtype_tp_name = "universal_dtypes.Posit32DType";
    static constexpr const char* dtype_attr = "Posit32DType";
    static constexpr const char* doc = "posit<32,2> scalar (tapered precision, NaR)";
};

struct Posit64Traits : PositTraitsBase<64, uint64_t> {
    static constexpr const char* name = "posit64";
    static constexpr const char* scalar_tp_name = "universal_dtypes.posit64";
    static constexpr const char* dtype_tp_name = "universal_dtypes.Posit64DType";
    static constexpr const char* dtype_attr = "Posit64DType";
    static constexpr const char* doc = "posit<64,2> scalar (tapered precision, NaR)";
};

}  // namespace

void register_posits(nb::module_& m) {
    // Each standard size is one line — the templated-type -> dtype-family harness
    // this issue set out to deliver. New configs (e.g. posit<10,1>) go here.
    register_universal_dtype<Posit8Traits>(m);
    register_universal_dtype<Posit16Traits>(m);
    register_universal_dtype<Posit32Traits>(m);
    register_universal_dtype<Posit64Traits>(m);
}
