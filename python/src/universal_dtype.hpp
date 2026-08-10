// universal_dtype.hpp — a reusable NEP-42 harness that binds a Universal C++
// number type to a NumPy dtype (issue #7).
//
// The bfloat16 work (#3) proved the NEP-42 registration path against a single
// concrete type. This header lifts that machinery into a template parameterized
// by a small `Traits` struct, so registering a new Universal type as a NumPy
// dtype is a one-liner:
//
//     struct Posit16Traits { ... };                 // ~10 lines of type facts
//     register_universal_dtype<Posit16Traits>(m);   // one line
//
// The templated types (posit<...> #7, cfloat #8, lns #9) all register through
// this. Everything type-specific lives in Traits:
//
//     using cpp_t;                 // the Universal C++ type (e.g. posit<16,2,uint16_t>)
//     using storage_t;             // the raw element storage (uintN_t, N = bit width)
//     static const char* name;     // "posit16"   (numpy dtype / scalar name)
//     static const char* scalar_tp_name;  // "universal_dtypes.posit16"
//     static const char* dtype_tp_name;   // "universal_dtypes.Posit16DType"
//     static const char* doc;
//     static storage_t   to_bits(const cpp_t&);   // raw encoding
//     static cpp_t       from_bits(storage_t);    // reconstruct from raw encoding
//     static cpp_t       from_double(double);      // round a real into the type
//     static double      to_double(const cpp_t&);  // convert back to double
//     static bool        is_nan(const cpp_t&);     // IEEE NaN / posit NaR
//     static bool        is_inf(const cpp_t&);     // IEEE inf (false if none, e.g. posit)
//     static bool        lt(const cpp_t&, const cpp_t&);  // ordering (full precision)
//     static bool        eq(const cpp_t&, const cpp_t&);  // equality (full precision)
//     static bool        is_zero(const cpp_t&);            // true for the zero value
//
// For a type whose to_double() is lossless (bfloat16/posit/cfloat/lns), lt/eq/
// is_zero are simply the to_double comparisons. A wide type (dd/…) must implement
// them with the C++ type's own operators so comparison/sort keep full precision.
//
// Arithmetic and comparisons are sourced from cpp_t's own operators, so the
// dtype's numerics are exactly the Universal type's — the whole point of the
// harness. Comparisons and nonzero go through to_double(), which maps the
// exceptional value (NaN / NaR) onto NaN so unordered semantics fall out
// naturally.

#pragma once

#include <Python.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <numpy/arrayobject.h>
#include <numpy/dtype_api.h>
#include <numpy/halffloat.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

#include <nanobind/nanobind.h>

#include <type_traits>

namespace nb = nanobind;

// A dtype's out-cast-to-IEEE-float safety. SAFE when to_double() is lossless (the
// small types), UNSAFE when it loses precision (dd and the other cascades). It is
// detected from an optional `Traits::to_float_casting`, defaulting to SAFE, so
// existing traits need no change and a wide type opts in with one line.
template <typename T, typename = void>
struct out_float_casting {
    static constexpr NPY_CASTING value = NPY_SAFE_CASTING;
};
template <typename T>
struct out_float_casting<T, std::void_t<decltype(T::to_float_casting)>> {
    static constexpr NPY_CASTING value = T::to_float_casting;
};

// ---- cross-dtype casts (issue #39) -----------------------------------------
// NumPy does not chain casts between two custom dtypes, so every ordered pair
// needs an explicit cast. Rather than an N*(N-1) matrix of templated loops, all
// cross-casts share ONE resolver and ONE strided loop; the (src, dst) identity
// is recovered at run time from the context descriptors via a small registry.
//
// The conversion is value-domain: src bits -> a K-term expansion of doubles ->
// dst bits. Each term is the residual of the previous, computed in the SOURCE
// type's own arithmetic (`v - from_double(to_double(v))`), so up to K*53 bits
// survive. `double` alone is insufficient for the wide configs (posit64, lns32,
// and the dd/td/qd cascades, all of which carry more than double's 53 bits); the
// expansion is exactly the compensated-sum / floating-point-expansion fallback
// the design mandates, built from each type's own operators. K = 4 covers even
// the widest registered type (qd_cascade, ~212 bits).
constexpr int UD_WIDE_TERMS = 4;

using UDToWide = void (*)(const char* in, double* w);     // src bits -> expansion
using UDFromWide = void (*)(char* out, const double* w);  // expansion -> dst bits

struct UDCrossEntry {
    PyArray_DTypeMeta* meta;
    UDToWide to_wide;
    UDFromWide from_wide;
};

// One registry shared across translation units (inline function's local static).
// Each dtype appends itself during registration; the loop looks entries up by
// their DTypeMeta.
inline std::vector<UDCrossEntry>& ud_cross_registry() {
    static std::vector<UDCrossEntry> reg;
    return reg;
}
inline const UDCrossEntry* ud_cross_find(PyArray_DTypeMeta* m) {
    for (const auto& e : ud_cross_registry())
        if (e.meta == m) return &e;
    return nullptr;
}

// Shared resolver: echo the given descriptors (or fall back to each side's
// default) and report the safety. Every cross-cast is UNSAFE — value-domain
// rounding is possible between different number systems.
inline NPY_CASTING ud_cross_resolve(PyObject*, PyArray_DTypeMeta* const dtypes[2],
                                    PyArray_Descr* const given[2], PyArray_Descr* loop[2],
                                    npy_intp*) {
    for (int i = 0; i < 2; i++) {
        if (given[i] != nullptr) { Py_INCREF(given[i]); loop[i] = given[i]; }
        else { loop[i] = PyArray_GetDefaultDescr(dtypes[i]); if (!loop[i]) return (NPY_CASTING)-1; }
    }
    return NPY_UNSAFE_CASTING;
}

// Shared strided loop for every cross-cast; the pair identity comes from the
// context descriptors, so one loop serves all N*(N-1) pairs.
inline int ud_cross_loop(PyArrayMethod_Context* ctx, char* const data[],
                         npy_intp const dims[], npy_intp const strides[], NpyAuxData*) {
    const UDCrossEntry* se = ud_cross_find(NPY_DTYPE(ctx->descriptors[0]));
    const UDCrossEntry* de = ud_cross_find(NPY_DTYPE(ctx->descriptors[1]));
    if (!se || !de) {
        PyErr_SetString(PyExc_RuntimeError, "universal_dtypes: unregistered cross-cast");
        return -1;
    }
    npy_intp N = dims[0];
    char* in = data[0];
    char* out = data[1];
    double w[UD_WIDE_TERMS];
    while (N--) {
        se->to_wide(in, w);
        de->from_wide(out, w);
        in += strides[0];
        out += strides[1];
    }
    return 0;
}

// Build one cross-cast spec. Per the DType API, NULL in `dtypes` means "the
// newly created DType", i.e. the one currently registering; the other side is a
// concrete, already-registered DTypeMeta.
inline PyArrayMethod_Spec* ud_make_cross_cast(PyArray_DTypeMeta* src, PyArray_DTypeMeta* dst) {
    auto** dts = (PyArray_DTypeMeta**)malloc(2 * sizeof(PyArray_DTypeMeta*));
    dts[0] = src; dts[1] = dst;
    auto* slots = (PyType_Slot*)malloc(4 * sizeof(PyType_Slot));
    slots[0] = {NPY_METH_resolve_descriptors, (void*)&ud_cross_resolve};
    slots[1] = {NPY_METH_strided_loop, (void*)&ud_cross_loop};
    slots[2] = {NPY_METH_unaligned_strided_loop, (void*)&ud_cross_loop};
    slots[3] = {0, nullptr};
    auto* spec = (PyArrayMethod_Spec*)malloc(sizeof(PyArrayMethod_Spec));
    spec->name = "ud_cross_cast"; spec->nin = 1; spec->nout = 1;
    spec->casting = NPY_UNSAFE_CASTING; spec->flags = NPY_METH_SUPPORTS_UNALIGNED;
    spec->dtypes = dts; spec->slots = slots;
    return spec;
}

// All per-type state (scalar type, DTypeMeta, cast specs) lives in static
// members of this class template, so each Traits gets its own independent set.
template <typename Traits>
struct UniversalDType {
    using cpp_t = typename Traits::cpp_t;
    using storage_t = typename Traits::storage_t;
    static constexpr int ELSIZE = static_cast<int>(sizeof(storage_t));

    // ---- Python scalar type -------------------------------------------------
    struct Scalar {
        PyObject_HEAD
        storage_t bits;
    };

    inline static PyTypeObject scalar_type{};
    inline static PyNumberMethods as_number{};
    inline static PyArray_DTypeMeta DType{};

    static inline int is_scalar(PyObject* obj) {
        return PyObject_TypeCheck(obj, &scalar_type);
    }

    static PyObject* scalar_from_bits(storage_t bits) {
        Scalar* o = PyObject_New(Scalar, &scalar_type);
        if (o) o->bits = bits;
        return reinterpret_cast<PyObject*>(o);
    }

    // Convert an arbitrary Python object to raw bits. Returns 0 on success.
    static int bits_from_pyobject(PyObject* obj, storage_t* out) {
        if (is_scalar(obj)) {
            *out = reinterpret_cast<Scalar*>(obj)->bits;
            return 0;
        }
        double d = PyFloat_AsDouble(obj);  // handles float, int, __float__
        if (d == -1.0 && PyErr_Occurred()) return -1;
        *out = Traits::to_bits(Traits::from_double(d));
        return 0;
    }

    static PyObject* scalar_new(PyTypeObject*, PyObject* args, PyObject*) {
        PyObject* value = nullptr;
        if (!PyArg_ParseTuple(args, "|O", &value)) return nullptr;
        storage_t bits = Traits::to_bits(Traits::from_double(0.0));
        if (value && bits_from_pyobject(value, &bits) < 0) return nullptr;
        return scalar_from_bits(bits);
    }

    static double scalar_value(PyObject* self) {
        return Traits::to_double(Traits::from_bits(reinterpret_cast<Scalar*>(self)->bits));
    }

    static PyObject* scalar_float(PyObject* self) {
        return PyFloat_FromDouble(scalar_value(self));
    }

    // Hash of the scalar's value. Scalar equality (scalar_richcompare) is defined
    // on the double value, so hashing that same value keeps hash consistent with
    // __eq__ (the data-model requirement) and matches Python's float/int hashing,
    // so a scalar and the equal `float`/`int` share a hash — as NumPy scalars do.
    static Py_hash_t scalar_hash(PyObject* self) {
        PyObject* f = PyFloat_FromDouble(scalar_value(self));
        if (!f) return -1;
        Py_hash_t h = PyObject_Hash(f);
        Py_DECREF(f);
        return h;
    }

    static PyObject* scalar_repr(PyObject* self) {
        PyObject* pf = PyFloat_FromDouble(scalar_value(self));
        if (!pf) return nullptr;
        PyObject* r = PyUnicode_FromFormat("%R", pf);
        Py_DECREF(pf);
        return r;
    }

    static PyObject* scalar_richcompare(PyObject* a, PyObject* b, int op) {
        double x, y;
        if (is_scalar(a)) x = scalar_value(a);
        else { x = PyFloat_AsDouble(a); if (x == -1.0 && PyErr_Occurred()) Py_RETURN_NOTIMPLEMENTED; }
        if (is_scalar(b)) y = scalar_value(b);
        else { y = PyFloat_AsDouble(b); if (y == -1.0 && PyErr_Occurred()) Py_RETURN_NOTIMPLEMENTED; }
        bool r = false;
        switch (op) {
            case Py_LT: r = x < y; break;
            case Py_LE: r = x <= y; break;
            case Py_EQ: r = x == y; break;
            case Py_NE: r = x != y; break;
            case Py_GT: r = x > y; break;
            case Py_GE: r = x >= y; break;
        }
        if (r) Py_RETURN_TRUE;
        Py_RETURN_FALSE;
    }

    // Pickle: a scalar reconstructs from its double value. This roundtrips
    // exactly — the stored value is representable, double holds it losslessly,
    // and constructing back rounds to the same value.
    static PyObject* scalar_reduce(PyObject* self, PyObject*) {
        return Py_BuildValue("(O(d))", (PyObject*)&scalar_type, scalar_value(self));
    }

    inline static PyMethodDef scalar_methods[2] = {
        {"__reduce__", (PyCFunction)scalar_reduce, METH_NOARGS, "pickle support"},
        {nullptr, nullptr, 0, nullptr},
    };

    // ---- descriptor helpers -------------------------------------------------
    static PyArray_Descr* new_descr() {
        PyArray_Descr* d = (PyArray_Descr*)PyArrayDescr_Type.tp_new(
            (PyTypeObject*)&DType, nullptr, nullptr);
        if (!d) return nullptr;
        d->elsize = ELSIZE;
        d->alignment = alignof(storage_t);
        return d;
    }

    static PyArray_Descr* canonical() {
        Py_INCREF(DType.singleton);
        return DType.singleton;
    }

    static PyObject* descr_repr(PyObject*) {
        return PyUnicode_FromFormat("dtype(%s)", Traits::name);
    }
    static PyObject* descr_str(PyObject*) {
        return PyUnicode_FromString(Traits::name);
    }
    static PyObject* descr_new(PyTypeObject*, PyObject*, PyObject*) {
        if (DType.singleton != nullptr) return (PyObject*)canonical();
        return (PyObject*)new_descr();
    }

    // Pickle for the dtype itself (needed to pickle arrays / np.save): reconstruct
    // via np.dtype(<scalar type>). Reconstructing by scalar type rather than by
    // string name is unambiguous even when another package registers the same
    // name (e.g. ml_dtypes also owns "bfloat16"); the scalar type pickles by
    // reference as universal_dtypes.<name>.
    static PyObject* descr_reduce(PyObject*, PyObject*) {
        PyObject* np = PyImport_ImportModule("numpy");
        if (!np) return nullptr;
        PyObject* dtype_callable = PyObject_GetAttrString(np, "dtype");
        Py_DECREF(np);
        if (!dtype_callable) return nullptr;
        return Py_BuildValue("(N(O))", dtype_callable, (PyObject*)&scalar_type);
    }

    inline static PyMethodDef descr_methods[2] = {
        {"__reduce__", (PyCFunction)descr_reduce, METH_NOARGS, "pickle support"},
        {nullptr, nullptr, 0, nullptr},
    };

    // ---- DType slots --------------------------------------------------------
    static PyArray_Descr* slot_default_descr(PyArray_DTypeMeta*) {
        if (DType.singleton != nullptr) return canonical();
        return new_descr();
    }

    static PyArray_DTypeMeta* slot_common_dtype(PyArray_DTypeMeta* cls, PyArray_DTypeMeta* other) {
        // The abstract DTypes standing in for weak Python scalars (NEP 50) are
        // not "numeric" by type_num — they have none — so they need naming
        // explicitly. Without this, np.result_type(a, 2) and anything built on
        // it (np.where, np.choose, ...) raise DTypePromotionError even though
        // the ufunc promoters resolve `a * 2` fine. Python complex is left out
        // deliberately: absorbing it would silently drop the imaginary part.
        if (other == &PyArray_PyLongDType || other == &PyArray_PyFloatDType) {
            Py_INCREF(cls);
            return cls;
        }
        // Promote against simple builtin numeric types (not complex/longdouble):
        // our type wins (values round into it). Everything else is NotImplemented.
        if (other->type_num >= 0 && PyTypeNum_ISNUMBER(other->type_num) &&
            !PyTypeNum_ISCOMPLEX(other->type_num) && other != &PyArray_LongDoubleDType) {
            Py_INCREF(cls);
            return cls;
        }
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta*)Py_NotImplemented;
    }

    static PyArray_Descr* slot_common_instance(PyArray_Descr*, PyArray_Descr*) { return canonical(); }
    static PyArray_Descr* slot_ensure_canonical(PyArray_Descr*) { return canonical(); }
    static PyArray_Descr* slot_discover(PyArray_DTypeMeta*, PyObject*) { return canonical(); }

    static int slot_setitem(PyArray_Descr*, PyObject* obj, char* dataptr) {
        storage_t bits;
        if (bits_from_pyobject(obj, &bits) < 0) return -1;
        std::memcpy(dataptr, &bits, ELSIZE);
        return 0;
    }

    static PyObject* slot_getitem(PyArray_Descr*, char* dataptr) {
        storage_t bits;
        std::memcpy(&bits, dataptr, ELSIZE);
        return scalar_from_bits(bits);
    }

    static int slot_compare(const void* a, const void* b, void*) {
        storage_t ba, bb;
        std::memcpy(&ba, a, ELSIZE);
        std::memcpy(&bb, b, ELSIZE);
        cpp_t va = Traits::from_bits(ba), vb = Traits::from_bits(bb);
        if (Traits::lt(va, vb)) return -1;
        if (Traits::lt(vb, va)) return 1;
        return 0;
    }

    static npy_bool slot_nonzero(void* data, void*) {
        storage_t bits;
        std::memcpy(&bits, data, ELSIZE);
        return Traits::is_zero(Traits::from_bits(bits)) ? NPY_FALSE : NPY_TRUE;
    }

    inline static PyType_Slot dtype_slots[10] = {
        {NPY_DT_default_descr, (void*)&slot_default_descr},
        {NPY_DT_common_dtype, (void*)&slot_common_dtype},
        {NPY_DT_common_instance, (void*)&slot_common_instance},
        {NPY_DT_ensure_canonical, (void*)&slot_ensure_canonical},
        {NPY_DT_discover_descr_from_pyobject, (void*)&slot_discover},
        {NPY_DT_setitem, (void*)&slot_setitem},
        {NPY_DT_getitem, (void*)&slot_getitem},
        {NPY_DT_PyArray_ArrFuncs_compare, (void*)&slot_compare},
        {NPY_DT_PyArray_ArrFuncs_nonzero, (void*)&slot_nonzero},
        {0, nullptr},
    };

    // ---- casts (all memcpy-based → unaligned-safe) --------------------------
    static int cast_self(PyArrayMethod_Context*, char* const data[], npy_intp const dims[],
                         npy_intp const strides[], NpyAuxData*) {
        npy_intp N = dims[0];
        char* in = data[0];
        char* out = data[1];
        while (N--) {
            std::memcpy(out, in, ELSIZE);
            in += strides[0];
            out += strides[1];
        }
        return 0;
    }

    static NPY_CASTING self_resolve(PyObject*, PyArray_DTypeMeta* const[2],
                                    PyArray_Descr* const given[2], PyArray_Descr* loop[2],
                                    npy_intp* view_offset) {
        Py_INCREF(given[0]);
        loop[0] = given[0];
        if (given[1] == nullptr) loop[1] = canonical();
        else { Py_INCREF(given[1]); loop[1] = given[1]; }
        *view_offset = 0;
        return NPY_NO_CASTING;
    }

    template <typename B>
    static int cast_to_builtin(PyArrayMethod_Context*, char* const data[], npy_intp const dims[],
                               npy_intp const strides[], NpyAuxData*) {
        npy_intp N = dims[0];
        char* in = data[0];
        char* out = data[1];
        while (N--) {
            storage_t bits;
            std::memcpy(&bits, in, ELSIZE);
            B v = static_cast<B>(Traits::to_double(Traits::from_bits(bits)));
            std::memcpy(out, &v, sizeof(B));
            in += strides[0];
            out += strides[1];
        }
        return 0;
    }

    template <typename B>
    static int cast_from_builtin(PyArrayMethod_Context*, char* const data[], npy_intp const dims[],
                                 npy_intp const strides[], NpyAuxData*) {
        npy_intp N = dims[0];
        char* in = data[0];
        char* out = data[1];
        while (N--) {
            B v;
            std::memcpy(&v, in, sizeof(B));
            storage_t bits = Traits::to_bits(Traits::from_double(static_cast<double>(v)));
            std::memcpy(out, &bits, ELSIZE);
            in += strides[0];
            out += strides[1];
        }
        return 0;
    }

    // ---- cross-dtype cast hooks (issue #39) ---------------------------------
    // Decompose one element into a K-term double expansion. Each term is the
    // residual of the previous, taken in cpp_t's own arithmetic, so the type's
    // full precision survives even when it exceeds double's 53 bits. NaN/inf are
    // carried in a single term (their residual is not meaningful).
    static void to_wide(const char* in, double* w) {
        storage_t bits;
        std::memcpy(&bits, in, ELSIZE);
        cpp_t v = Traits::from_bits(bits);
        double d0 = Traits::to_double(v);
        if (!std::isfinite(d0)) {
            w[0] = d0;
            for (int k = 1; k < UD_WIDE_TERMS; k++) w[k] = 0.0;
            return;
        }
        for (int k = 0; k < UD_WIDE_TERMS; k++) {
            double d = Traits::to_double(v);
            w[k] = d;
            v = v - Traits::from_double(d);  // residual in the type's own arithmetic
        }
    }

    // Reconstruct an element from a K-term expansion by summing the terms in
    // cpp_t's arithmetic, smallest first (a compensated sum), then rounding into
    // the type. A NaN/inf head term is rounded straight in.
    static void from_wide(char* out, const double* w) {
        cpp_t acc;
        if (!std::isfinite(w[0])) {
            acc = Traits::from_double(w[0]);
        } else {
            acc = Traits::from_double(0.0);
            for (int k = UD_WIDE_TERMS - 1; k >= 0; k--)
                acc = acc + Traits::from_double(w[k]);
        }
        storage_t rb = Traits::to_bits(acc);
        std::memcpy(out, &rb, ELSIZE);
    }

    // float16 needs dedicated loops: npy_half is a 16-bit bit pattern, not a C
    // floating type, so the cast_to_builtin/from_builtin static_cast path would
    // corrupt it. Convert through the value domain with numpy's half helpers.
    static int cast_to_half(PyArrayMethod_Context*, char* const data[], npy_intp const dims[],
                            npy_intp const strides[], NpyAuxData*) {
        npy_intp N = dims[0];
        char *in = data[0], *out = data[1];
        while (N--) {
            storage_t bits;
            std::memcpy(&bits, in, ELSIZE);
            npy_half h = npy_double_to_half(Traits::to_double(Traits::from_bits(bits)));
            std::memcpy(out, &h, sizeof(npy_half));
            in += strides[0]; out += strides[1];
        }
        return 0;
    }
    static int cast_from_half(PyArrayMethod_Context*, char* const data[], npy_intp const dims[],
                              npy_intp const strides[], NpyAuxData*) {
        npy_intp N = dims[0];
        char *in = data[0], *out = data[1];
        while (N--) {
            npy_half h;
            std::memcpy(&h, in, sizeof(npy_half));
            storage_t bits = Traits::to_bits(Traits::from_double(npy_half_to_double(h)));
            std::memcpy(out, &bits, ELSIZE);
            in += strides[0]; out += strides[1];
        }
        return 0;
    }

    static NPY_CASTING builtin_resolve(PyObject*, PyArray_DTypeMeta* const dtypes[2],
                                       PyArray_Descr* const given[2], PyArray_Descr* loop[2],
                                       npy_intp*) {
        for (int i = 0; i < 2; i++) {
            if (given[i] != nullptr) { Py_INCREF(given[i]); loop[i] = given[i]; }
            else if (dtypes[i] == &DType) loop[i] = canonical();
            else { loop[i] = PyArray_GetDefaultDescr(dtypes[i]); if (!loop[i]) return (NPY_CASTING)-1; }
        }
        return NPY_UNSAFE_CASTING;  // lossy; astype still works
    }

    inline static PyArray_DTypeMeta* self_dtypes[2] = {&DType, &DType};
    inline static PyType_Slot self_slots[4] = {
        {NPY_METH_resolve_descriptors, (void*)&self_resolve},
        {NPY_METH_strided_loop, (void*)&cast_self},
        {NPY_METH_unaligned_strided_loop, (void*)&cast_self},
        {0, nullptr},
    };
    inline static PyArrayMethod_Spec self_cast{};
    inline static std::vector<PyArrayMethod_Spec*> casts;

    static PyArrayMethod_Spec* make_cast(const char* name, PyArray_DTypeMeta* src,
                                         PyArray_DTypeMeta* dst, PyArrayMethod_StridedLoop* loop,
                                         NPY_CASTING casting) {
        auto** dts = (PyArray_DTypeMeta**)malloc(2 * sizeof(PyArray_DTypeMeta*));
        dts[0] = src; dts[1] = dst;
        auto* slots = (PyType_Slot*)malloc(4 * sizeof(PyType_Slot));
        slots[0] = {NPY_METH_resolve_descriptors, (void*)&builtin_resolve};
        slots[1] = {NPY_METH_strided_loop, (void*)loop};
        slots[2] = {NPY_METH_unaligned_strided_loop, (void*)loop};
        slots[3] = {0, nullptr};
        auto* spec = (PyArrayMethod_Spec*)malloc(sizeof(PyArrayMethod_Spec));
        spec->name = name; spec->nin = 1; spec->nout = 1; spec->casting = casting;
        spec->flags = NPY_METH_SUPPORTS_UNALIGNED; spec->dtypes = dts; spec->slots = slots;
        return spec;
    }

    static PyArrayMethod_Spec** build_casts() {
        self_cast.name = "cast_self";
        self_cast.nin = 1; self_cast.nout = 1;
        self_cast.casting = NPY_NO_CASTING; self_cast.flags = NPY_METH_SUPPORTS_UNALIGNED;
        self_cast.dtypes = self_dtypes; self_cast.slots = self_slots;

        casts.clear();
        casts.push_back(&self_cast);
        constexpr NPY_CASTING out_float = out_float_casting<Traits>::value;
        casts.push_back(make_cast("to_float", &DType, &PyArray_FloatDType,
                                  (PyArrayMethod_StridedLoop*)&cast_to_builtin<float>, out_float));
        casts.push_back(make_cast("to_double", &DType, &PyArray_DoubleDType,
                                  (PyArrayMethod_StridedLoop*)&cast_to_builtin<double>, out_float));
        casts.push_back(make_cast("to_longlong", &DType, &PyArray_LongLongDType,
                                  (PyArrayMethod_StridedLoop*)&cast_to_builtin<long long>, NPY_UNSAFE_CASTING));
        // Inbound float/int stay UNSAFE: rounding a whole float64 array into a
        // low-precision type is a data-loss decision the caller should make
        // explicitly with .astype(). Weak Python scalars do NOT need this
        // relaxed — common_dtype names their abstract DTypes, so NumPy builds
        // the scalar directly in this dtype instead of casting an int64/float64
        // temporary (#55).
        casts.push_back(make_cast("from_float", &PyArray_FloatDType, &DType,
                                  (PyArrayMethod_StridedLoop*)&cast_from_builtin<float>, NPY_UNSAFE_CASTING));
        casts.push_back(make_cast("from_double", &PyArray_DoubleDType, &DType,
                                  (PyArrayMethod_StridedLoop*)&cast_from_builtin<double>, NPY_UNSAFE_CASTING));
        casts.push_back(make_cast("from_longlong", &PyArray_LongLongDType, &DType,
                                  (PyArrayMethod_StridedLoop*)&cast_from_builtin<long long>, NPY_UNSAFE_CASTING));
        casts.push_back(make_cast("from_long", &PyArray_LongDType, &DType,
                                  (PyArrayMethod_StridedLoop*)&cast_from_builtin<long>, NPY_UNSAFE_CASTING));
        // bool is the exception, and the only cast level this change touches.
        // Python `True` is not a weak scalar — NumPy maps it straight to the
        // concrete BoolDType — so `a * True` really does cast bool -> this type,
        // and a ufunc checks its inputs at same_kind. NumPy grades bool -> any
        // numeric as `safe`; SAME_KIND is the honest grade here only because the
        // ±1 fractional formats saturate 1.0 to maxpos.
        casts.push_back(make_cast("from_bool", &PyArray_BoolDType, &DType,
                                  (PyArrayMethod_StridedLoop*)&cast_from_builtin<npy_bool>,
                                  NPY_SAME_KIND_CASTING));
        // float16 both directions UNSAFE — half's 11-bit significand is lossy.
        casts.push_back(make_cast("to_half", &DType, &PyArray_HalfDType,
                                  (PyArrayMethod_StridedLoop*)&cast_to_half, NPY_UNSAFE_CASTING));
        casts.push_back(make_cast("from_half", &PyArray_HalfDType, &DType,
                                  (PyArrayMethod_StridedLoop*)&cast_from_half, NPY_UNSAFE_CASTING));

        // Cross-dtype casts to/from every universal dtype registered before this
        // one (all fully initialized). This dtype's own side is NULL ("the newly
        // created DType"). Placing each pair in the later-registered dtype's spec
        // means every ordered pair is registered exactly once, with no circular
        // dependency on a not-yet-initialized DTypeMeta.
        for (const auto& e : ud_cross_registry()) {
            casts.push_back(ud_make_cross_cast(nullptr, e.meta));  // this -> e
            casts.push_back(ud_make_cross_cast(e.meta, nullptr));  // e -> this
        }
        casts.push_back(nullptr);
        return casts.data();
    }

    // ---- ufunc loops --------------------------------------------------------
    // numpy passes exactly nin+nout descriptors with NO terminator, so the
    // operand count is a compile-time parameter (reading past it corrupts memory).
    template <int NARGS>
    static NPY_CASTING ufunc_resolve(PyObject*, PyArray_DTypeMeta* const dtypes[],
                                     PyArray_Descr* const given[], PyArray_Descr* loop[],
                                     npy_intp*) {
        for (int i = 0; i < NARGS; i++) {
            if (given[i] != nullptr) { Py_INCREF(given[i]); loop[i] = given[i]; }
            else if (dtypes[i] == &DType) loop[i] = canonical();
            else { loop[i] = PyArray_GetDefaultDescr(dtypes[i]); if (!loop[i]) return (NPY_CASTING)-1; }
        }
        return NPY_NO_CASTING;
    }

    // Arithmetic sourced from cpp_t's own operators.
    template <cpp_t (*Op)(cpp_t, cpp_t)>
    static int binary_loop(PyArrayMethod_Context*, char* const data[], npy_intp const dims[],
                           npy_intp const strides[], NpyAuxData*) {
        npy_intp N = dims[0];
        char *i0 = data[0], *i1 = data[1], *o = data[2];
        while (N--) {
            storage_t a, b;
            std::memcpy(&a, i0, ELSIZE);
            std::memcpy(&b, i1, ELSIZE);
            storage_t r = Traits::to_bits(Op(Traits::from_bits(a), Traits::from_bits(b)));
            std::memcpy(o, &r, ELSIZE);
            i0 += strides[0]; i1 += strides[1]; o += strides[2];
        }
        return 0;
    }

    template <cpp_t (*Op)(cpp_t)>
    static int unary_loop(PyArrayMethod_Context*, char* const data[], npy_intp const dims[],
                          npy_intp const strides[], NpyAuxData*) {
        npy_intp N = dims[0];
        char *i0 = data[0], *o = data[1];
        while (N--) {
            storage_t a;
            std::memcpy(&a, i0, ELSIZE);
            storage_t r = Traits::to_bits(Op(Traits::from_bits(a)));
            std::memcpy(o, &r, ELSIZE);
            i0 += strides[0]; o += strides[1];
        }
        return 0;
    }

    // Comparisons run on cpp_t values via the trait's lt/eq, so a type whose
    // to_double is lossy (e.g. dd's ~106-bit value) compares at full precision.
    template <bool (*Cmp)(const cpp_t&, const cpp_t&)>
    static int cmp_loop(PyArrayMethod_Context*, char* const data[], npy_intp const dims[],
                        npy_intp const strides[], NpyAuxData*) {
        npy_intp N = dims[0];
        char *i0 = data[0], *i1 = data[1], *o = data[2];
        while (N--) {
            storage_t a, b;
            std::memcpy(&a, i0, ELSIZE);
            std::memcpy(&b, i1, ELSIZE);
            npy_bool r = Cmp(Traits::from_bits(a), Traits::from_bits(b)) ? NPY_TRUE : NPY_FALSE;
            std::memcpy(o, &r, sizeof(npy_bool));
            i0 += strides[0]; i1 += strides[1]; o += strides[2];
        }
        return 0;
    }

    // Special-value predicates. is_nan is the type's NaN/NaR; is_inf is the
    // type's infinity (false for types without one, e.g. posit); isfinite is the
    // complement of both. Each writes an npy_bool output.
    template <bool (*Pred)(const cpp_t&)>
    static int predicate_loop(PyArrayMethod_Context*, char* const data[], npy_intp const dims[],
                              npy_intp const strides[], NpyAuxData*) {
        npy_intp N = dims[0];
        char *i0 = data[0], *o = data[1];
        while (N--) {
            storage_t a;
            std::memcpy(&a, i0, ELSIZE);
            npy_bool r = Pred(Traits::from_bits(a)) ? NPY_TRUE : NPY_FALSE;
            std::memcpy(o, &r, sizeof(npy_bool));
            i0 += strides[0]; o += strides[1];
        }
        return 0;
    }
    // clip is a dedicated 3-in ufunc in NumPy 2.x (not composed from
    // minimum/maximum), so it needs its own loop: clip(a, lo, hi) bounds a into
    // [lo, hi] at full precision. NaN in a propagates; a NaN bound is ignored.
    static int clip_loop(PyArrayMethod_Context*, char* const data[], npy_intp const dims[],
                         npy_intp const strides[], NpyAuxData*) {
        npy_intp N = dims[0];
        char *ia = data[0], *ilo = data[1], *ihi = data[2], *o = data[3];
        while (N--) {
            storage_t A, LO, HI;
            std::memcpy(&A, ia, ELSIZE);
            std::memcpy(&LO, ilo, ELSIZE);
            std::memcpy(&HI, ihi, ELSIZE);
            cpp_t r = Traits::from_bits(A);
            if (!Traits::is_nan(r)) {
                cpp_t lo = Traits::from_bits(LO), hi = Traits::from_bits(HI);
                if (!Traits::is_nan(lo) && Traits::lt(r, lo)) r = lo;
                if (!Traits::is_nan(hi) && Traits::lt(hi, r)) r = hi;
            }
            storage_t rb = Traits::to_bits(r);
            std::memcpy(o, &rb, ELSIZE);
            ia += strides[0]; ilo += strides[1]; ihi += strides[2]; o += strides[3];
        }
        return 0;
    }

    static bool pred_isnan(const cpp_t& v) { return Traits::is_nan(v); }
    static bool pred_isinf(const cpp_t& v) { return Traits::is_inf(v); }
    static bool pred_isfinite(const cpp_t& v) { return !Traits::is_nan(v) && !Traits::is_inf(v); }

    static cpp_t op_add(cpp_t a, cpp_t b) { return a + b; }
    static cpp_t op_sub(cpp_t a, cpp_t b) { return a - b; }
    static cpp_t op_mul(cpp_t a, cpp_t b) { return a * b; }
    static cpp_t op_div(cpp_t a, cpp_t b) { return a / b; }
    static cpp_t op_neg(cpp_t a) { return -a; }
    static cpp_t op_abs(cpp_t a) { return Traits::to_double(a) < 0.0 ? cpp_t(-a) : a; }
    // minimum/maximum compare at full precision (via the trait's lt) and, like
    // NumPy's, PROPAGATE NaN/NaR: if either operand is NaN the result is NaN.
    static cpp_t op_max(cpp_t a, cpp_t b) {
        if (Traits::is_nan(a)) return a;
        if (Traits::is_nan(b)) return b;
        return Traits::lt(a, b) ? b : a;
    }
    static cpp_t op_min(cpp_t a, cpp_t b) {
        if (Traits::is_nan(a)) return a;
        if (Traits::is_nan(b)) return b;
        return Traits::lt(a, b) ? a : b;
    }
    // fmax/fmin instead SUPPRESS NaN: a NaN operand is ignored (matches NumPy,
    // and gives np.nanmax/np.nanmin their behaviour).
    static cpp_t op_fmax(cpp_t a, cpp_t b) {
        if (Traits::is_nan(a)) return b;
        if (Traits::is_nan(b)) return a;
        return Traits::lt(a, b) ? b : a;
    }
    static cpp_t op_fmin(cpp_t a, cpp_t b) {
        if (Traits::is_nan(a)) return b;
        if (Traits::is_nan(b)) return a;
        return Traits::lt(a, b) ? a : b;
    }
    // All six comparisons derive from the trait's lt/eq. NaN/NaR are unordered:
    // lt and eq both return false, so <,<=,>,>=,== are false and != is true.
    static bool cmp_eq(const cpp_t& a, const cpp_t& b) { return Traits::eq(a, b); }
    static bool cmp_ne(const cpp_t& a, const cpp_t& b) { return !Traits::eq(a, b); }
    static bool cmp_lt(const cpp_t& a, const cpp_t& b) { return Traits::lt(a, b); }
    static bool cmp_le(const cpp_t& a, const cpp_t& b) { return Traits::lt(a, b) || Traits::eq(a, b); }
    static bool cmp_gt(const cpp_t& a, const cpp_t& b) { return Traits::lt(b, a); }
    static bool cmp_ge(const cpp_t& a, const cpp_t& b) { return Traits::lt(b, a) || Traits::eq(a, b); }

    // Unary math ufuncs (exp/log/trig/sqrt/...): compute in double, round back
    // into the type. Higher-precision-then-round is the correct semantics for a
    // low-precision type and keeps one implementation across every Traits.
    template <double (*Fn)(double)>
    static int math_loop(PyArrayMethod_Context*, char* const data[], npy_intp const dims[],
                         npy_intp const strides[], NpyAuxData*) {
        npy_intp N = dims[0];
        char *i0 = data[0], *o = data[1];
        while (N--) {
            storage_t a;
            std::memcpy(&a, i0, ELSIZE);
            double r = Fn(Traits::to_double(Traits::from_bits(a)));
            storage_t rb = Traits::to_bits(Traits::from_double(r));
            std::memcpy(o, &rb, ELSIZE);
            i0 += strides[0]; o += strides[1];
        }
        return 0;
    }

    // Binary math ufuncs (power): compute in double, round back into the type —
    // same higher-precision-then-round rationale as the unary math loop.
    template <double (*Fn)(double, double)>
    static int math2_loop(PyArrayMethod_Context*, char* const data[], npy_intp const dims[],
                          npy_intp const strides[], NpyAuxData*) {
        npy_intp N = dims[0];
        char *i0 = data[0], *i1 = data[1], *o = data[2];
        while (N--) {
            storage_t a, b;
            std::memcpy(&a, i0, ELSIZE);
            std::memcpy(&b, i1, ELSIZE);
            double r = Fn(Traits::to_double(Traits::from_bits(a)),
                          Traits::to_double(Traits::from_bits(b)));
            storage_t rb = Traits::to_bits(Traits::from_double(r));
            std::memcpy(o, &rb, ELSIZE);
            i0 += strides[0]; i1 += strides[1]; o += strides[2];
        }
        return 0;
    }
    static double m2_pow(double a, double b) { return std::pow(a, b); }

#define UDT_MATH1(nm, expr) static double nm(double x) { return (expr); }
    UDT_MATH1(m_sqrt, std::sqrt(x))
    UDT_MATH1(m_cbrt, std::cbrt(x))
    UDT_MATH1(m_square, x* x)
    UDT_MATH1(m_recip, 1.0 / x)
    UDT_MATH1(m_exp, std::exp(x))
    UDT_MATH1(m_exp2, std::exp2(x))
    UDT_MATH1(m_expm1, std::expm1(x))
    UDT_MATH1(m_log, std::log(x))
    UDT_MATH1(m_log2, std::log2(x))
    UDT_MATH1(m_log10, std::log10(x))
    UDT_MATH1(m_log1p, std::log1p(x))
    UDT_MATH1(m_sin, std::sin(x))
    UDT_MATH1(m_cos, std::cos(x))
    UDT_MATH1(m_tan, std::tan(x))
    UDT_MATH1(m_asin, std::asin(x))
    UDT_MATH1(m_acos, std::acos(x))
    UDT_MATH1(m_atan, std::atan(x))
    UDT_MATH1(m_sinh, std::sinh(x))
    UDT_MATH1(m_cosh, std::cosh(x))
    UDT_MATH1(m_tanh, std::tanh(x))
    UDT_MATH1(m_floor, std::floor(x))
    UDT_MATH1(m_ceil, std::ceil(x))
    UDT_MATH1(m_trunc, std::trunc(x))
    UDT_MATH1(m_rint, std::rint(x))
    UDT_MATH1(m_sign, std::isnan(x) ? x : static_cast<double>((x > 0.0) - (x < 0.0)))
#undef UDT_MATH1

    static PyObject* get_ufunc(const char* name) {
        PyObject* np = PyImport_ImportModule("numpy");
        if (!np) return nullptr;
        PyObject* uf = PyObject_GetAttrString(np, name);
        Py_DECREF(np);
        if (uf && PyObject_TypeCheck(uf, &PyUFunc_Type)) return uf;
        // A few ufuncs (e.g. clip) are shadowed at the top level by a Python
        // dispatcher; the ufunc object itself lives in numpy._core.umath.
        Py_XDECREF(uf);
        PyErr_Clear();
        PyObject* um = PyImport_ImportModule("numpy._core.umath");
        if (!um) { PyErr_Clear(); um = PyImport_ImportModule("numpy.core.umath"); }
        if (!um) return nullptr;
        uf = PyObject_GetAttrString(um, name);
        Py_DECREF(um);
        return uf;
    }

    // Reduction identities, so an empty sum/prod returns the identity (matching
    // NumPy: sum([]) -> 0, prod([]) -> 1) instead of raising. min/max
    // deliberately have none — NumPy itself raises on an empty min/max.
    static int reduce_init_zero(PyArrayMethod_Context*, npy_bool, void* initial) {
        storage_t b = Traits::to_bits(Traits::from_double(0.0));
        std::memcpy(initial, &b, ELSIZE);
        return 1;
    }
    static int reduce_init_one(PyArrayMethod_Context*, npy_bool, void* initial) {
        storage_t b = Traits::to_bits(Traits::from_double(1.0));
        std::memcpy(initial, &b, ELSIZE);
        return 1;
    }

    // ---- promoters: let Python int/float operands adopt this dtype ----------
    // NumPy 2 dispatches a ufunc on the *exact* DType signature of its operands.
    // A Python scalar arrives as one of the abstract weak DTypes (PyLongDType /
    // PyFloatDType, per NEP 50), which matches no registered loop — so `a * 2`
    // raises UFuncTypeError even though promote_types(this, float64) already
    // says this type wins. A promoter rewrites such a mixed signature to the
    // all-this-type one, so dispatch lands on the ordinary (T, T) -> T loop and
    // the scalar is converted through setitem (bits_from_pyobject). See #55.
    //
    // Only Python-scalar operands are absorbed (the two weak DTypes, plus
    // concrete bool — see add_scalar_promoters). A concrete numeric builtin
    // (np.float64(2), a float64 array) is deliberately left to raise: silently
    // rounding a whole float64 array into a low-precision type is a data-loss
    // decision the caller should make explicitly with .astype().
    template <int NIN, bool BOOL_OUT>
    static int promote_to_self(PyObject*, PyArray_DTypeMeta* const[],
                               PyArray_DTypeMeta* const signature[],
                               PyArray_DTypeMeta* new_op_dtypes[]) {
        for (int i = 0; i < NIN; i++) {
            Py_INCREF(&DType);
            new_op_dtypes[i] = &DType;
        }
        // Respect an explicit output dtype from signature=/dtype=; otherwise the
        // natural one (bool for comparisons, this type for arithmetic).
        PyArray_DTypeMeta* out = signature[NIN] ? signature[NIN]
                                 : BOOL_OUT     ? &PyArray_BoolDType
                                                : &DType;
        Py_INCREF(out);
        new_op_dtypes[NIN] = out;
        return 0;
    }

    static int add_promoter(const char* ufunc_name, PyArray_DTypeMeta* const dts[], int nargs,
                            PyArrayMethod_PromoterFunction* fn) {
        PyObject* ufunc = get_ufunc(ufunc_name);
        if (!ufunc) return -1;
        PyObject* tup = PyTuple_New(nargs);
        if (!tup) { Py_DECREF(ufunc); return -1; }
        for (int i = 0; i < nargs; i++) {
            Py_INCREF((PyObject*)dts[i]);
            PyTuple_SET_ITEM(tup, i, (PyObject*)dts[i]);
        }
        PyObject* cap = PyCapsule_New((void*)fn, "numpy._ufunc_promoter", nullptr);
        if (!cap) { Py_DECREF(tup); Py_DECREF(ufunc); return -1; }
        int r = PyUFunc_AddPromoter(ufunc, tup, cap);
        Py_DECREF(cap);
        Py_DECREF(tup);
        Py_DECREF(ufunc);
        return r;
    }

    // Register a promoter for every mixed signature of `ufunc_name` in which at
    // least one operand is this dtype and the others are absorbable scalars.
    // Requiring at least one operand of this dtype matters: a promoter is
    // registered globally on the ufunc, so an all-scalar signature would hijack
    // plain `2 * 3` for every user of NumPy.
    //
    // Absorbable = the two weak Python scalar DTypes, plus concrete bool.
    // Python `True` is not weak — NumPy maps it straight to BoolDType — but
    // bool -> numeric carries no information loss (it is `safe` in NumPy's own
    // table), so absorbing it does not hide a lossy conversion the way a
    // float64 operand would.
    static int add_scalar_promoters(const char* ufunc_name, int nin, bool bool_out) {
        PyArray_DTypeMeta* absorbable[3] = {&PyArray_PyLongDType, &PyArray_PyFloatDType,
                                            &PyArray_BoolDType};
        PyArrayMethod_PromoterFunction* fn =
            (nin == 3)  ? (PyArrayMethod_PromoterFunction*)&promote_to_self<3, false>
            : bool_out  ? (PyArrayMethod_PromoterFunction*)&promote_to_self<2, true>
                        : (PyArrayMethod_PromoterFunction*)&promote_to_self<2, false>;
        constexpr int NKIND = 4;  // this dtype + the 3 absorbable ones
        int ncombo = 1;
        for (int i = 0; i < nin; i++) ncombo *= NKIND;
        for (int c = 0; c < ncombo; c++) {
            PyArray_DTypeMeta* dts[4];
            int code = c, nself = 0;
            for (int i = 0; i < nin; i++) {
                int k = code % NKIND;
                code /= NKIND;
                if (k == 0) { dts[i] = &DType; nself++; }
                else { dts[i] = absorbable[k - 1]; }
            }
            // nself == nin is the real loop; nself == 0 would steal other dtypes'
            // dispatch. Only the genuinely mixed signatures get a promoter.
            if (nself == 0 || nself == nin) continue;
            dts[nin] = bool_out ? &PyArray_BoolDType : &DType;
            if (add_promoter(ufunc_name, dts, nin + 1, fn) < 0) return -1;
        }
        return 0;
    }

    static int add_ufunc(const char* ufunc_name, PyArray_DTypeMeta** dtypes, int nin, int nout,
                         PyArrayMethod_StridedLoop* loop,
                         PyArrayMethod_GetReductionInitial* initial = nullptr) {
        PyObject* ufunc = get_ufunc(ufunc_name);
        if (!ufunc) return -1;
        void* resolve = (nin + nout == 2)   ? (void*)&ufunc_resolve<2>
                        : (nin + nout == 4) ? (void*)&ufunc_resolve<4>
                                            : (void*)&ufunc_resolve<3>;
        PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, resolve},
            {NPY_METH_strided_loop, (void*)loop},
            {NPY_METH_unaligned_strided_loop, (void*)loop},
            {0, nullptr},  // optional NPY_METH_get_reduction_initial
            {0, nullptr},
        };
        if (initial) slots[3] = {NPY_METH_get_reduction_initial, (void*)initial};
        PyArrayMethod_Spec spec;
        spec.name = ufunc_name; spec.nin = nin; spec.nout = nout;
        spec.casting = NPY_NO_CASTING; spec.flags = NPY_METH_SUPPORTS_UNALIGNED;
        spec.dtypes = dtypes; spec.slots = slots;
        int r = PyUFunc_AddLoopFromSpec(ufunc, &spec);
        Py_DECREF(ufunc);
        return r;
    }

    static int init_ufuncs() {
        PyArray_DTypeMeta* ttt[3] = {&DType, &DType, &DType};
        PyArray_DTypeMeta* tt[2] = {&DType, &DType};
        PyArray_DTypeMeta* tto[3] = {&DType, &DType, &PyArray_BoolDType};
        PyArray_DTypeMeta* to[2] = {&DType, &PyArray_BoolDType};

        if (add_ufunc("add", ttt, 2, 1, (PyArrayMethod_StridedLoop*)&binary_loop<op_add>,
                      &reduce_init_zero)) return -1;  // sum([]) -> 0
        if (add_ufunc("subtract", ttt, 2, 1, (PyArrayMethod_StridedLoop*)&binary_loop<op_sub>)) return -1;
        if (add_ufunc("multiply", ttt, 2, 1, (PyArrayMethod_StridedLoop*)&binary_loop<op_mul>,
                      &reduce_init_one)) return -1;  // prod([]) -> 1
        if (add_ufunc("true_divide", ttt, 2, 1, (PyArrayMethod_StridedLoop*)&binary_loop<op_div>)) return -1;
        // power computes in double then rounds back (like the unary math ufuncs).
        if (add_ufunc("power", ttt, 2, 1, (PyArrayMethod_StridedLoop*)&math2_loop<m2_pow>)) return -1;
        // minimum/maximum propagate NaN; fmin/fmax suppress it (drives np.min/max,
        // np.clip, np.nanmin/nanmax). All compare at full precision via lt.
        if (add_ufunc("minimum", ttt, 2, 1, (PyArrayMethod_StridedLoop*)&binary_loop<op_min>)) return -1;
        if (add_ufunc("maximum", ttt, 2, 1, (PyArrayMethod_StridedLoop*)&binary_loop<op_max>)) return -1;
        if (add_ufunc("fmin", ttt, 2, 1, (PyArrayMethod_StridedLoop*)&binary_loop<op_fmin>)) return -1;
        if (add_ufunc("fmax", ttt, 2, 1, (PyArrayMethod_StridedLoop*)&binary_loop<op_fmax>)) return -1;
        // clip is its own 3-in ufunc in NumPy 2.x (a, lo, hi) -> out.
        PyArray_DTypeMeta* tttt[4] = {&DType, &DType, &DType, &DType};
        if (add_ufunc("clip", tttt, 3, 1, (PyArrayMethod_StridedLoop*)&clip_loop)) return -1;
        if (add_ufunc("negative", tt, 1, 1, (PyArrayMethod_StridedLoop*)&unary_loop<op_neg>)) return -1;
        if (add_ufunc("absolute", tt, 1, 1, (PyArrayMethod_StridedLoop*)&unary_loop<op_abs>)) return -1;
        if (add_ufunc("isnan", to, 1, 1, (PyArrayMethod_StridedLoop*)&predicate_loop<pred_isnan>))
            return -1;
        if (add_ufunc("isinf", to, 1, 1, (PyArrayMethod_StridedLoop*)&predicate_loop<pred_isinf>))
            return -1;
        if (add_ufunc("isfinite", to, 1, 1,
                      (PyArrayMethod_StridedLoop*)&predicate_loop<pred_isfinite>))
            return -1;
        if (add_ufunc("equal", tto, 2, 1, (PyArrayMethod_StridedLoop*)&cmp_loop<cmp_eq>)) return -1;
        if (add_ufunc("not_equal", tto, 2, 1, (PyArrayMethod_StridedLoop*)&cmp_loop<cmp_ne>)) return -1;
        if (add_ufunc("less", tto, 2, 1, (PyArrayMethod_StridedLoop*)&cmp_loop<cmp_lt>)) return -1;
        if (add_ufunc("less_equal", tto, 2, 1, (PyArrayMethod_StridedLoop*)&cmp_loop<cmp_le>)) return -1;
        if (add_ufunc("greater", tto, 2, 1, (PyArrayMethod_StridedLoop*)&cmp_loop<cmp_gt>)) return -1;
        if (add_ufunc("greater_equal", tto, 2, 1, (PyArrayMethod_StridedLoop*)&cmp_loop<cmp_ge>)) return -1;

        // unary math ufuncs (in -> out, both this type)
        struct M { const char* name; PyArrayMethod_StridedLoop* loop; };
        const M maths[] = {
            {"sqrt", (PyArrayMethod_StridedLoop*)&math_loop<m_sqrt>},
            {"cbrt", (PyArrayMethod_StridedLoop*)&math_loop<m_cbrt>},
            {"square", (PyArrayMethod_StridedLoop*)&math_loop<m_square>},
            {"reciprocal", (PyArrayMethod_StridedLoop*)&math_loop<m_recip>},
            {"exp", (PyArrayMethod_StridedLoop*)&math_loop<m_exp>},
            {"exp2", (PyArrayMethod_StridedLoop*)&math_loop<m_exp2>},
            {"expm1", (PyArrayMethod_StridedLoop*)&math_loop<m_expm1>},
            {"log", (PyArrayMethod_StridedLoop*)&math_loop<m_log>},
            {"log2", (PyArrayMethod_StridedLoop*)&math_loop<m_log2>},
            {"log10", (PyArrayMethod_StridedLoop*)&math_loop<m_log10>},
            {"log1p", (PyArrayMethod_StridedLoop*)&math_loop<m_log1p>},
            {"sin", (PyArrayMethod_StridedLoop*)&math_loop<m_sin>},
            {"cos", (PyArrayMethod_StridedLoop*)&math_loop<m_cos>},
            {"tan", (PyArrayMethod_StridedLoop*)&math_loop<m_tan>},
            {"arcsin", (PyArrayMethod_StridedLoop*)&math_loop<m_asin>},
            {"arccos", (PyArrayMethod_StridedLoop*)&math_loop<m_acos>},
            {"arctan", (PyArrayMethod_StridedLoop*)&math_loop<m_atan>},
            {"sinh", (PyArrayMethod_StridedLoop*)&math_loop<m_sinh>},
            {"cosh", (PyArrayMethod_StridedLoop*)&math_loop<m_cosh>},
            {"tanh", (PyArrayMethod_StridedLoop*)&math_loop<m_tanh>},
            {"floor", (PyArrayMethod_StridedLoop*)&math_loop<m_floor>},
            {"ceil", (PyArrayMethod_StridedLoop*)&math_loop<m_ceil>},
            {"trunc", (PyArrayMethod_StridedLoop*)&math_loop<m_trunc>},
            {"rint", (PyArrayMethod_StridedLoop*)&math_loop<m_rint>},
            {"sign", (PyArrayMethod_StridedLoop*)&math_loop<m_sign>},
        };
        PyArray_DTypeMeta* tt2[2] = {&DType, &DType};
        for (const M& mm : maths) {
            if (add_ufunc(mm.name, tt2, 1, 1, mm.loop)) return -1;
        }

        // Promoters for the mixed (this dtype, Python scalar) signatures — see
        // add_scalar_promoters. Unary ufuncs need none (no second operand).
        static const char* const binary_arith[] = {"add",     "subtract", "multiply",
                                                   "true_divide", "power", "minimum",
                                                   "maximum", "fmin",     "fmax"};
        for (const char* name : binary_arith) {
            if (add_scalar_promoters(name, 2, false)) return -1;
        }
        static const char* const comparisons[] = {"equal", "not_equal",    "less",
                                                  "less_equal", "greater", "greater_equal"};
        for (const char* name : comparisons) {
            if (add_scalar_promoters(name, 2, true)) return -1;
        }
        if (add_scalar_promoters("clip", 3, false)) return -1;
        return 0;
    }

    // ---- registration -------------------------------------------------------
    static void register_(nb::module_& m) {
        // scalar type
        scalar_type = {PyVarObject_HEAD_INIT(nullptr, 0)};
        scalar_type.tp_name = Traits::scalar_tp_name;
        scalar_type.tp_basicsize = sizeof(Scalar);
        scalar_type.tp_flags = Py_TPFLAGS_DEFAULT;
        scalar_type.tp_doc = Traits::doc;
        scalar_type.tp_new = scalar_new;
        scalar_type.tp_repr = scalar_repr;
        scalar_type.tp_richcompare = scalar_richcompare;
        scalar_type.tp_hash = scalar_hash;
        scalar_type.tp_methods = scalar_methods;
        as_number.nb_float = scalar_float;
        scalar_type.tp_as_number = &as_number;
        if (PyType_Ready(&scalar_type) < 0)
            throw std::runtime_error(std::string(Traits::name) + " scalar type not ready");

        // DTypeMeta (runtime-initialized metaclass instance — the C
        // designated-init-after-HEAD idiom does not compile in C++)
        std::memset(&DType, 0, sizeof(DType));
        PyObject* dtobj = (PyObject*)&DType;
        Py_SET_REFCNT(dtobj, 1);
        Py_SET_TYPE(dtobj, &PyArrayDTypeMeta_Type);
        PyTypeObject* dt = (PyTypeObject*)&DType;
        dt->tp_name = Traits::dtype_tp_name;
        dt->tp_base = &PyArrayDescr_Type;
        dt->tp_basicsize = sizeof(PyArray_Descr);
        dt->tp_flags = Py_TPFLAGS_DEFAULT;
        dt->tp_repr = descr_repr;
        dt->tp_str = descr_str;
        dt->tp_new = descr_new;
        dt->tp_methods = descr_methods;
        if (PyType_Ready(dt) < 0)
            throw std::runtime_error(std::string(Traits::name) + " DType not ready");

        PyArrayDTypeMeta_Spec spec;
        spec.typeobj = &scalar_type;
        spec.flags = NPY_DT_NUMERIC;
        spec.casts = build_casts();
        spec.slots = dtype_slots;
        spec.baseclass = nullptr;
        if (PyArrayInitDTypeMeta_FromSpec(&DType, &spec) < 0) {
            PyErr_Print();
            throw std::runtime_error(std::string(Traits::name) + " DType FromSpec failed");
        }

        if (DType.singleton == nullptr) DType.singleton = PyArray_GetDefaultDescr(&DType);
        if (DType.singleton == nullptr) {
            if (PyErr_Occurred()) PyErr_Print();
            throw std::runtime_error(std::string(Traits::name) + " default descriptor failed");
        }

        for (int i = 1; casts[i] != nullptr; i++) {  // slot 0 is the static self-cast
            free(casts[i]->dtypes);
            free(casts[i]->slots);
            free(casts[i]);
        }

        if (init_ufuncs() < 0)
            throw std::runtime_error(std::string(Traits::name) + " ufunc registration failed");

        // np.dtype(<scalar>) resolves via spec.typeobj; make np.dtype("<name>")
        // resolve too by registering the scalar type in numpy's sctypeDict.
        register_string_name();

        // Publish this dtype so (a) later dtypes wire cross-casts to/from it and
        // (b) the shared cross-cast loop can resolve it at run time.
        ud_cross_registry().push_back({&DType, &to_wide, &from_wide});

        m.attr(Traits::name) = nb::borrow(reinterpret_cast<PyObject*>(&scalar_type));
        m.attr(Traits::dtype_attr) = nb::borrow(reinterpret_cast<PyObject*>(&DType));
    }

    // Make np.dtype("<name>") resolve to this DType. numpy checks its scalar-type
    // dictionary (`numpy._core.sctypeDict` / `numpy.core.sctypeDict`) when given a
    // string, so we add the scalar type under its short name.
    static void register_string_name() {
        PyObject* np = PyImport_ImportModule("numpy");
        if (!np) { PyErr_Clear(); return; }
        PyObject* d = PyObject_GetAttrString(np, "sctypeDict");
        Py_DECREF(np);
        if (!d) { PyErr_Clear(); return; }
        // Don't clobber a name another package already owns (e.g. ml_dtypes also
        // registers "bfloat16"). Pickling doesn't depend on this — it goes
        // through the scalar type — so leaving an existing owner in place is safe.
        if (PyDict_GetItemString(d, Traits::name) == nullptr) {
            PyDict_SetItemString(d, Traits::name, (PyObject*)&scalar_type);
        }
        Py_DECREF(d);
    }
};

template <typename Traits>
void register_universal_dtype(nb::module_& m) {
    UniversalDType<Traits>::register_(m);
}
