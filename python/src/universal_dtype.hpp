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

#include <numpy/arrayobject.h>
#include <numpy/dtype_api.h>
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
    inline static PyArrayMethod_Spec* casts[16];

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

        int n = 0;
        casts[n++] = &self_cast;
        constexpr NPY_CASTING out_float = out_float_casting<Traits>::value;
        casts[n++] = make_cast("to_float", &DType, &PyArray_FloatDType,
                               (PyArrayMethod_StridedLoop*)&cast_to_builtin<float>, out_float);
        casts[n++] = make_cast("to_double", &DType, &PyArray_DoubleDType,
                               (PyArrayMethod_StridedLoop*)&cast_to_builtin<double>, out_float);
        casts[n++] = make_cast("to_longlong", &DType, &PyArray_LongLongDType,
                               (PyArrayMethod_StridedLoop*)&cast_to_builtin<long long>, NPY_UNSAFE_CASTING);
        casts[n++] = make_cast("from_float", &PyArray_FloatDType, &DType,
                               (PyArrayMethod_StridedLoop*)&cast_from_builtin<float>, NPY_UNSAFE_CASTING);
        casts[n++] = make_cast("from_double", &PyArray_DoubleDType, &DType,
                               (PyArrayMethod_StridedLoop*)&cast_from_builtin<double>, NPY_UNSAFE_CASTING);
        casts[n++] = make_cast("from_longlong", &PyArray_LongLongDType, &DType,
                               (PyArrayMethod_StridedLoop*)&cast_from_builtin<long long>, NPY_UNSAFE_CASTING);
        casts[n++] = make_cast("from_long", &PyArray_LongDType, &DType,
                               (PyArrayMethod_StridedLoop*)&cast_from_builtin<long>, NPY_UNSAFE_CASTING);
        casts[n++] = make_cast("from_bool", &PyArray_BoolDType, &DType,
                               (PyArrayMethod_StridedLoop*)&cast_from_builtin<npy_bool>, NPY_UNSAFE_CASTING);
        casts[n] = nullptr;
        return casts;
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
    static bool pred_isnan(const cpp_t& v) { return Traits::is_nan(v); }
    static bool pred_isinf(const cpp_t& v) { return Traits::is_inf(v); }
    static bool pred_isfinite(const cpp_t& v) { return !Traits::is_nan(v) && !Traits::is_inf(v); }

    static cpp_t op_add(cpp_t a, cpp_t b) { return a + b; }
    static cpp_t op_sub(cpp_t a, cpp_t b) { return a - b; }
    static cpp_t op_mul(cpp_t a, cpp_t b) { return a * b; }
    static cpp_t op_div(cpp_t a, cpp_t b) { return a / b; }
    static cpp_t op_neg(cpp_t a) { return -a; }
    static cpp_t op_abs(cpp_t a) { return Traits::to_double(a) < 0.0 ? cpp_t(-a) : a; }
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
        return uf;
    }

    static int add_ufunc(const char* ufunc_name, PyArray_DTypeMeta** dtypes, int nin, int nout,
                         PyArrayMethod_StridedLoop* loop) {
        PyObject* ufunc = get_ufunc(ufunc_name);
        if (!ufunc) return -1;
        void* resolve = (nin + nout == 2) ? (void*)&ufunc_resolve<2> : (void*)&ufunc_resolve<3>;
        PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, resolve},
            {NPY_METH_strided_loop, (void*)loop},
            {NPY_METH_unaligned_strided_loop, (void*)loop},
            {0, nullptr},
        };
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

        if (add_ufunc("add", ttt, 2, 1, (PyArrayMethod_StridedLoop*)&binary_loop<op_add>)) return -1;
        if (add_ufunc("subtract", ttt, 2, 1, (PyArrayMethod_StridedLoop*)&binary_loop<op_sub>)) return -1;
        if (add_ufunc("multiply", ttt, 2, 1, (PyArrayMethod_StridedLoop*)&binary_loop<op_mul>)) return -1;
        if (add_ufunc("true_divide", ttt, 2, 1, (PyArrayMethod_StridedLoop*)&binary_loop<op_div>)) return -1;
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
