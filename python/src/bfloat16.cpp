// bfloat16 NumPy dtype — NEP-42 DType API (NumPy 2.x only), issue #3.
//
// This is the modern, non-legacy registration path: bfloat16 is a real
// PyArray_DTypeMeta (a subclass of np.dtype) registered via
// PyArrayInitDTypeMeta_FromSpec, with casts and ufunc loops implemented as
// ArrayMethods. bf16<->float32 use canonical round-to-nearest-even, which
// matches ml_dtypes.bfloat16 bit-for-bit; arithmetic computes in float32 and
// rounds to bf16 (the correct bf16 semantics). Routing the round through
// Universal's bfloat16 is a follow-up commit — numerically identical, but it
// establishes the "dtype delegates to a Universal C++ type" harness that the
// templated types (posit<...>) will reuse.
//
// Modeled on NumPy's own NEP-42 examples: numpy-user-dtypes/metadatadtype
// (registration skeleton) and quaddtype (custom ufunc ArrayMethods).

#include <Python.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#define PY_ARRAY_UNIQUE_SYMBOL universal_dtypes_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL universal_dtypes_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/dtype_api.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

#include <universal/number/bfloat16/bfloat16.hpp>

#include <nanobind/nanobind.h>
namespace nb = nanobind;

// --------------------------------------------------------------------------
// bf16 numerics are delegated to Universal's sw::universal::bfloat16 — the same
// C++ type used across the Universal library. bfloat16(float) rounds
// round-to-nearest-even and operator float() converts back; both are bit-for-bit
// identical to the previous hand-rolled conversion (still validated by the
// ml_dtypes oracle). Delegating here — plus the arithmetic loops below, which
// use Universal's own operators — establishes the "dtype delegates to a
// Universal C++ type" harness that the templated types (posit<...>) reuse.
// --------------------------------------------------------------------------
using ubfloat16 = sw::universal::bfloat16;

static inline float bf16_bits_to_float(uint16_t h) {
    ubfloat16 b;
    b.setbits(h);
    return float(b);
}

static inline uint16_t float_to_bf16_bits(float f) {
    return ubfloat16(f).bits();
}

// --------------------------------------------------------------------------
// Python scalar type: universal_dtypes.bfloat16
// --------------------------------------------------------------------------
typedef struct {
    PyObject_HEAD
    uint16_t bits;
} PyBfloat16Object;

static PyTypeObject PyBfloat16_Type;

static inline int is_bfloat16_scalar(PyObject* obj) {
    return PyObject_TypeCheck(obj, &PyBfloat16_Type);
}

static PyObject* bfloat16_from_bits(uint16_t bits) {
    PyBfloat16Object* o = PyObject_New(PyBfloat16Object, &PyBfloat16_Type);
    if (o) o->bits = bits;
    return reinterpret_cast<PyObject*>(o);
}

// Convert an arbitrary Python object to bf16 bits. Returns 0 on success.
static int bfloat16_bits_from_pyobject(PyObject* obj, uint16_t* out) {
    if (is_bfloat16_scalar(obj)) {
        *out = reinterpret_cast<PyBfloat16Object*>(obj)->bits;
        return 0;
    }
    double d = PyFloat_AsDouble(obj);  // handles float, int, and __float__
    if (d == -1.0 && PyErr_Occurred()) return -1;
    *out = float_to_bf16_bits(static_cast<float>(d));
    return 0;
}

static PyObject* PyBfloat16_New(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    (void)type;
    (void)kwds;
    PyObject* value = nullptr;
    if (!PyArg_ParseTuple(args, "|O", &value)) return nullptr;
    uint16_t bits = 0;  // bfloat16() == 0.0
    if (value && bfloat16_bits_from_pyobject(value, &bits) < 0) return nullptr;
    return bfloat16_from_bits(bits);
}

static PyObject* PyBfloat16_Float(PyObject* self) {
    return PyFloat_FromDouble(
        static_cast<double>(bf16_bits_to_float(reinterpret_cast<PyBfloat16Object*>(self)->bits)));
}

static PyObject* PyBfloat16_Repr(PyObject* self) {
    float f = bf16_bits_to_float(reinterpret_cast<PyBfloat16Object*>(self)->bits);
    PyObject* pf = PyFloat_FromDouble(static_cast<double>(f));
    if (!pf) return nullptr;
    PyObject* r = PyUnicode_FromFormat("%R", pf);
    Py_DECREF(pf);
    return r;
}

static PyObject* PyBfloat16_RichCompare(PyObject* a, PyObject* b, int op) {
    double x, y;
    if (is_bfloat16_scalar(a)) x = bf16_bits_to_float(reinterpret_cast<PyBfloat16Object*>(a)->bits);
    else { x = PyFloat_AsDouble(a); if (x == -1.0 && PyErr_Occurred()) Py_RETURN_NOTIMPLEMENTED; }
    if (is_bfloat16_scalar(b)) y = bf16_bits_to_float(reinterpret_cast<PyBfloat16Object*>(b)->bits);
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

static PyNumberMethods bfloat16_as_number = {};

// --------------------------------------------------------------------------
// The DType (a PyArray_DTypeMeta — subclass of np.dtype). Its instances are
// plain PyArray_Descr; bf16 is non-parametric so there is a single canonical
// descriptor (the singleton).
// --------------------------------------------------------------------------
static PyArray_DTypeMeta Bfloat16DType;

static PyArray_Descr* canonical_bf16();  // forward decl

// repr/str for descriptor instances: np.dtype(bfloat16) -> dtype(bfloat16)
static PyObject* bf16_descr_repr(PyObject* NPY_UNUSED(self)) {
    return PyUnicode_FromString("dtype(bfloat16)");
}
static PyObject* bf16_descr_str(PyObject* NPY_UNUSED(self)) {
    return PyUnicode_FromString("bfloat16");
}

// Create a fresh descriptor instance (elsize=2).
static PyArray_Descr* new_bf16_descr() {
    PyArray_Descr* d = (PyArray_Descr*)PyArrayDescr_Type.tp_new(
        (PyTypeObject*)&Bfloat16DType, nullptr, nullptr);
    if (!d) return nullptr;
    d->elsize = sizeof(uint16_t);
    d->alignment = alignof(uint16_t);
    return d;
}

// User DTypes must define their own tp_new (numpy forbids inheriting
// np.dtype.__new__). bf16 is non-parametric, so any construction yields the
// canonical descriptor.
static PyObject* bf16_descr_new(PyTypeObject* NPY_UNUSED(subtype), PyObject* NPY_UNUSED(args),
                                PyObject* NPY_UNUSED(kwds)) {
    if (Bfloat16DType.singleton != nullptr) return (PyObject*)canonical_bf16();
    return (PyObject*)new_bf16_descr();
}

// Return a new reference to the canonical descriptor.
static PyArray_Descr* canonical_bf16() {
    Py_INCREF(Bfloat16DType.singleton);
    return Bfloat16DType.singleton;
}

// ---- DType slots ----------------------------------------------------------

static PyArray_Descr* bf16_default_descr(PyArray_DTypeMeta* NPY_UNUSED(cls)) {
    if (Bfloat16DType.singleton != nullptr) return canonical_bf16();
    return new_bf16_descr();
}

static PyArray_DTypeMeta* bf16_common_dtype(PyArray_DTypeMeta* cls, PyArray_DTypeMeta* other) {
    // Promote against simple builtin numeric types (not complex/longdouble):
    // bf16 wins (values round into bf16). Everything else is NotImplemented.
    if (other->type_num >= 0 && PyTypeNum_ISNUMBER(other->type_num) &&
        !PyTypeNum_ISCOMPLEX(other->type_num) && other != &PyArray_LongDoubleDType) {
        Py_INCREF(cls);
        return cls;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta*)Py_NotImplemented;
}

static PyArray_Descr* bf16_common_instance(PyArray_Descr* NPY_UNUSED(d1),
                                           PyArray_Descr* NPY_UNUSED(d2)) {
    return canonical_bf16();
}

static PyArray_Descr* bf16_ensure_canonical(PyArray_Descr* NPY_UNUSED(self)) {
    return canonical_bf16();
}

static PyArray_Descr* bf16_discover_descr_from_pyobject(PyArray_DTypeMeta* NPY_UNUSED(cls),
                                                        PyObject* NPY_UNUSED(obj)) {
    // Any scalar we accept maps to the single canonical descriptor.
    return canonical_bf16();
}

static int bf16_setitem(PyArray_Descr* NPY_UNUSED(descr), PyObject* obj, char* dataptr) {
    uint16_t bits;
    if (bfloat16_bits_from_pyobject(obj, &bits) < 0) return -1;
    std::memcpy(dataptr, &bits, sizeof(bits));
    return 0;
}

static PyObject* bf16_getitem(PyArray_Descr* NPY_UNUSED(descr), char* dataptr) {
    uint16_t bits;
    std::memcpy(&bits, dataptr, sizeof(bits));
    return bfloat16_from_bits(bits);
}

// Legacy ArrFuncs reachable as NEP-42 slots (compare/nonzero) — used by
// sort, nonzero, and boolean coercion.
static int bf16_compare(const void* a, const void* b, void* NPY_UNUSED(arr)) {
    uint16_t ba, bb;
    std::memcpy(&ba, a, 2);
    std::memcpy(&bb, b, 2);
    float fa = bf16_bits_to_float(ba), fb = bf16_bits_to_float(bb);
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

static npy_bool bf16_nonzero(void* data, void* NPY_UNUSED(arr)) {
    uint16_t bits;
    std::memcpy(&bits, data, 2);
    return bf16_bits_to_float(bits) != 0.0f ? NPY_TRUE : NPY_FALSE;
}

static PyType_Slot Bfloat16DType_Slots[] = {
    {NPY_DT_default_descr, (void*)&bf16_default_descr},
    {NPY_DT_common_dtype, (void*)&bf16_common_dtype},
    {NPY_DT_common_instance, (void*)&bf16_common_instance},
    {NPY_DT_ensure_canonical, (void*)&bf16_ensure_canonical},
    {NPY_DT_discover_descr_from_pyobject, (void*)&bf16_discover_descr_from_pyobject},
    {NPY_DT_setitem, (void*)&bf16_setitem},
    {NPY_DT_getitem, (void*)&bf16_getitem},
    {NPY_DT_PyArray_ArrFuncs_compare, (void*)&bf16_compare},
    {NPY_DT_PyArray_ArrFuncs_nonzero, (void*)&bf16_nonzero},
    {0, nullptr},
};

// --------------------------------------------------------------------------
// Casts. All loops are memcpy-based, so they are safe for unaligned data and
// serve as both the aligned and unaligned strided loop.
// --------------------------------------------------------------------------

// bf16 -> bf16 (canonical copy)
static int cast_bf16_to_bf16(PyArrayMethod_Context* NPY_UNUSED(ctx), char* const data[],
                             npy_intp const dims[], npy_intp const strides[],
                             NpyAuxData* NPY_UNUSED(ad)) {
    npy_intp N = dims[0];
    char* in = data[0];
    char* out = data[1];
    while (N--) {
        std::memcpy(out, in, 2);
        in += strides[0];
        out += strides[1];
    }
    return 0;
}

static NPY_CASTING bf16_to_bf16_resolve(PyObject* NPY_UNUSED(self),
                                        PyArray_DTypeMeta* const NPY_UNUSED(dtypes[2]),
                                        PyArray_Descr* const given_descrs[2],
                                        PyArray_Descr* loop_descrs[2], npy_intp* view_offset) {
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    if (given_descrs[1] == nullptr) {
        loop_descrs[1] = canonical_bf16();
    } else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }
    *view_offset = 0;  // identical representation
    return NPY_NO_CASTING;
}

// bf16 -> T
template <typename T>
static int cast_bf16_to_T(PyArrayMethod_Context* NPY_UNUSED(ctx), char* const data[],
                          npy_intp const dims[], npy_intp const strides[],
                          NpyAuxData* NPY_UNUSED(ad)) {
    npy_intp N = dims[0];
    char* in = data[0];
    char* out = data[1];
    while (N--) {
        uint16_t b;
        std::memcpy(&b, in, 2);
        T v = static_cast<T>(bf16_bits_to_float(b));
        std::memcpy(out, &v, sizeof(T));
        in += strides[0];
        out += strides[1];
    }
    return 0;
}

// T -> bf16
template <typename T>
static int cast_T_to_bf16(PyArrayMethod_Context* NPY_UNUSED(ctx), char* const data[],
                          npy_intp const dims[], npy_intp const strides[],
                          NpyAuxData* NPY_UNUSED(ad)) {
    npy_intp N = dims[0];
    char* in = data[0];
    char* out = data[1];
    while (N--) {
        T v;
        std::memcpy(&v, in, sizeof(T));
        uint16_t b = float_to_bf16_bits(static_cast<float>(v));
        std::memcpy(out, &b, 2);
        in += strides[0];
        out += strides[1];
    }
    return 0;
}

// Resolver for a bf16<->builtin cast. Places the canonical bf16 descriptor and
// a canonical builtin descriptor for whichever side is missing.
static NPY_CASTING bf16_builtin_resolve(PyObject* NPY_UNUSED(self),
                                        PyArray_DTypeMeta* const dtypes[2],
                                        PyArray_Descr* const given_descrs[2],
                                        PyArray_Descr* loop_descrs[2],
                                        npy_intp* NPY_UNUSED(view_offset)) {
    for (int i = 0; i < 2; i++) {
        if (given_descrs[i] != nullptr) {
            Py_INCREF(given_descrs[i]);
            loop_descrs[i] = given_descrs[i];
        } else if (dtypes[i] == &Bfloat16DType) {
            loop_descrs[i] = canonical_bf16();
        } else {
            loop_descrs[i] = PyArray_GetDefaultDescr(dtypes[i]);
            if (loop_descrs[i] == nullptr) return (NPY_CASTING)-1;
        }
    }
    return NPY_UNSAFE_CASTING;  // float/int <-> bf16 is lossy; astype still works
}

// Build one PyArrayMethod_Spec on the heap (freed after FromSpec).
static PyArrayMethod_Spec* make_cast_spec(const char* name, PyArray_DTypeMeta* src,
                                          PyArray_DTypeMeta* dst,
                                          PyArrayMethod_StridedLoop* loop,
                                          NPY_CASTING casting) {
    PyArray_DTypeMeta** dts = (PyArray_DTypeMeta**)malloc(2 * sizeof(PyArray_DTypeMeta*));
    dts[0] = src;
    dts[1] = dst;

    // memcpy-based loops are unaligned-safe, so the same fn serves both slots.
    PyType_Slot* slots = (PyType_Slot*)malloc(4 * sizeof(PyType_Slot));
    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = (void*)&bf16_builtin_resolve;
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = (void*)loop;
    slots[2].slot = NPY_METH_unaligned_strided_loop;
    slots[2].pfunc = (void*)loop;
    slots[3].slot = 0;
    slots[3].pfunc = nullptr;

    PyArrayMethod_Spec* spec = (PyArrayMethod_Spec*)malloc(sizeof(PyArrayMethod_Spec));
    spec->name = name;
    spec->nin = 1;
    spec->nout = 1;
    spec->casting = casting;
    spec->flags = NPY_METH_SUPPORTS_UNALIGNED;
    spec->dtypes = dts;
    spec->slots = slots;
    return spec;
}

// The within-dtype cast uses a static spec (NumPy requires it be present).
static PyArray_DTypeMeta* bf16_self_dtypes[2] = {&Bfloat16DType, &Bfloat16DType};
static PyType_Slot bf16_self_slots[] = {
    {NPY_METH_resolve_descriptors, (void*)&bf16_to_bf16_resolve},
    {NPY_METH_strided_loop, (void*)&cast_bf16_to_bf16},
    {NPY_METH_unaligned_strided_loop, (void*)&cast_bf16_to_bf16},
    {0, nullptr},
};
static PyArrayMethod_Spec Bfloat16ToBfloat16Cast = {
    /*.name=*/"cast_bfloat16_to_bfloat16",
    /*.nin=*/1,
    /*.nout=*/1,
    /*.casting=*/NPY_NO_CASTING,
    /*.flags=*/NPY_METH_SUPPORTS_UNALIGNED,
    /*.dtypes=*/bf16_self_dtypes,
    /*.slots=*/bf16_self_slots,
};

// Assembled at registration time (NULL-terminated). Slot 0 is the self-cast.
static PyArrayMethod_Spec* g_casts[16];

static PyArrayMethod_Spec** build_casts() {
    int n = 0;
    g_casts[n++] = &Bfloat16ToBfloat16Cast;
    // bf16 -> builtin
    g_casts[n++] = make_cast_spec("cast_bfloat16_to_float", &Bfloat16DType, &PyArray_FloatDType,
                                  (PyArrayMethod_StridedLoop*)&cast_bf16_to_T<float>,
                                  NPY_SAFE_CASTING);
    g_casts[n++] = make_cast_spec("cast_bfloat16_to_double", &Bfloat16DType, &PyArray_DoubleDType,
                                  (PyArrayMethod_StridedLoop*)&cast_bf16_to_T<double>,
                                  NPY_SAFE_CASTING);
    g_casts[n++] = make_cast_spec("cast_bfloat16_to_longlong", &Bfloat16DType,
                                  &PyArray_LongLongDType,
                                  (PyArrayMethod_StridedLoop*)&cast_bf16_to_T<long long>,
                                  NPY_UNSAFE_CASTING);
    // builtin -> bf16
    g_casts[n++] = make_cast_spec("cast_float_to_bfloat16", &PyArray_FloatDType, &Bfloat16DType,
                                  (PyArrayMethod_StridedLoop*)&cast_T_to_bf16<float>,
                                  NPY_UNSAFE_CASTING);
    g_casts[n++] = make_cast_spec("cast_double_to_bfloat16", &PyArray_DoubleDType, &Bfloat16DType,
                                  (PyArrayMethod_StridedLoop*)&cast_T_to_bf16<double>,
                                  NPY_UNSAFE_CASTING);
    g_casts[n++] = make_cast_spec("cast_longlong_to_bfloat16", &PyArray_LongLongDType,
                                  &Bfloat16DType,
                                  (PyArrayMethod_StridedLoop*)&cast_T_to_bf16<long long>,
                                  NPY_UNSAFE_CASTING);
    g_casts[n++] = make_cast_spec("cast_long_to_bfloat16", &PyArray_LongDType, &Bfloat16DType,
                                  (PyArrayMethod_StridedLoop*)&cast_T_to_bf16<long>,
                                  NPY_UNSAFE_CASTING);
    g_casts[n++] = make_cast_spec("cast_bool_to_bfloat16", &PyArray_BoolDType, &Bfloat16DType,
                                  (PyArrayMethod_StridedLoop*)&cast_T_to_bf16<npy_bool>,
                                  NPY_UNSAFE_CASTING);
    g_casts[n] = nullptr;
    return g_casts;
}

// --------------------------------------------------------------------------
// Ufunc loops (compute in float32, round to bf16). Registered as ArrayMethods
// via PyUFunc_AddLoopFromSpec.
// --------------------------------------------------------------------------

// Resolver: force canonical bf16 for bf16 operands; default builtin (bool)
// outputs. numpy passes exactly nin+nout descriptors with NO terminator, so the
// operand count is a compile-time parameter (reading past it corrupts the heap).
template <int NARGS>
static NPY_CASTING bf16_ufunc_resolve(PyObject* NPY_UNUSED(self),
                                      PyArray_DTypeMeta* const dtypes[],
                                      PyArray_Descr* const given_descrs[],
                                      PyArray_Descr* loop_descrs[],
                                      npy_intp* NPY_UNUSED(view_offset)) {
    for (int i = 0; i < NARGS; i++) {
        if (given_descrs[i] != nullptr) {
            Py_INCREF(given_descrs[i]);
            loop_descrs[i] = given_descrs[i];
        } else if (dtypes[i] == &Bfloat16DType) {
            loop_descrs[i] = canonical_bf16();
        } else {
            loop_descrs[i] = PyArray_GetDefaultDescr(dtypes[i]);
            if (loop_descrs[i] == nullptr) return (NPY_CASTING)-1;
        }
    }
    return NPY_NO_CASTING;
}

// Arithmetic runs on Universal's bfloat16: bits -> ubfloat16 -> Universal op ->
// bits. Universal's operators compute in float and round on assignment, so this
// matches ml_dtypes exactly while sourcing the numerics from the C++ type.
template <ubfloat16 (*Op)(ubfloat16, ubfloat16)>
static int bf16_binary_loop(PyArrayMethod_Context* NPY_UNUSED(ctx), char* const data[],
                            npy_intp const dims[], npy_intp const strides[],
                            NpyAuxData* NPY_UNUSED(ad)) {
    npy_intp N = dims[0];
    char *i0 = data[0], *i1 = data[1], *o = data[2];
    while (N--) {
        uint16_t a, b;
        std::memcpy(&a, i0, 2);
        std::memcpy(&b, i1, 2);
        ubfloat16 ua, ub;
        ua.setbits(a);
        ub.setbits(b);
        uint16_t r = Op(ua, ub).bits();
        std::memcpy(o, &r, 2);
        i0 += strides[0];
        i1 += strides[1];
        o += strides[2];
    }
    return 0;
}
static ubfloat16 op_add(ubfloat16 a, ubfloat16 b) { return a + b; }
static ubfloat16 op_sub(ubfloat16 a, ubfloat16 b) { return a - b; }
static ubfloat16 op_mul(ubfloat16 a, ubfloat16 b) { return a * b; }
static ubfloat16 op_div(ubfloat16 a, ubfloat16 b) { return a / b; }

template <ubfloat16 (*Op)(ubfloat16)>
static int bf16_unary_loop(PyArrayMethod_Context* NPY_UNUSED(ctx), char* const data[],
                           npy_intp const dims[], npy_intp const strides[],
                           NpyAuxData* NPY_UNUSED(ad)) {
    npy_intp N = dims[0];
    char *i0 = data[0], *o = data[1];
    while (N--) {
        uint16_t a;
        std::memcpy(&a, i0, 2);
        ubfloat16 ua;
        ua.setbits(a);
        uint16_t r = Op(ua).bits();
        std::memcpy(o, &r, 2);
        i0 += strides[0];
        o += strides[1];
    }
    return 0;
}
static ubfloat16 op_neg(ubfloat16 a) { return -a; }
static ubfloat16 op_abs(ubfloat16 a) { return a.isneg() ? -a : a; }

template <bool (*Cmp)(float, float)>
static int bf16_cmp_loop(PyArrayMethod_Context* NPY_UNUSED(ctx), char* const data[],
                         npy_intp const dims[], npy_intp const strides[],
                         NpyAuxData* NPY_UNUSED(ad)) {
    npy_intp N = dims[0];
    char *i0 = data[0], *i1 = data[1], *o = data[2];
    while (N--) {
        uint16_t a, b;
        std::memcpy(&a, i0, 2);
        std::memcpy(&b, i1, 2);
        npy_bool r = Cmp(bf16_bits_to_float(a), bf16_bits_to_float(b)) ? NPY_TRUE : NPY_FALSE;
        std::memcpy(o, &r, sizeof(npy_bool));
        i0 += strides[0];
        i1 += strides[1];
        o += strides[2];
    }
    return 0;
}
static bool cmp_eq(float a, float b) { return a == b; }
static bool cmp_ne(float a, float b) { return a != b; }
static bool cmp_lt(float a, float b) { return a < b; }
static bool cmp_le(float a, float b) { return a <= b; }
static bool cmp_gt(float a, float b) { return a > b; }
static bool cmp_ge(float a, float b) { return a >= b; }

static PyObject* get_ufunc(const char* name) {
    PyObject* np = PyImport_ImportModule("numpy");
    if (!np) return nullptr;
    PyObject* uf = PyObject_GetAttrString(np, name);
    Py_DECREF(np);
    return uf;
}

static int add_loop(const char* ufunc_name, PyArray_DTypeMeta** dtypes, int nin, int nout,
                    PyArrayMethod_StridedLoop* loop) {
    PyObject* ufunc = get_ufunc(ufunc_name);
    if (!ufunc) return -1;
    // Select the resolver by operand count — numpy passes exactly nin+nout
    // descriptors with no terminator, so the count must be known statically.
    void* resolve = (nin + nout == 2) ? (void*)&bf16_ufunc_resolve<2>
                                      : (void*)&bf16_ufunc_resolve<3>;
    PyType_Slot slots[] = {
        {NPY_METH_resolve_descriptors, resolve},
        {NPY_METH_strided_loop, (void*)loop},
        {NPY_METH_unaligned_strided_loop, (void*)loop},
        {0, nullptr},
    };
    PyArrayMethod_Spec spec;
    spec.name = ufunc_name;
    spec.nin = nin;
    spec.nout = nout;
    spec.casting = NPY_NO_CASTING;
    spec.flags = NPY_METH_SUPPORTS_UNALIGNED;
    spec.dtypes = dtypes;
    spec.slots = slots;
    int r = PyUFunc_AddLoopFromSpec(ufunc, &spec);
    Py_DECREF(ufunc);
    return r;
}

static int init_ufuncs() {
    PyArray_DTypeMeta* bbb[3] = {&Bfloat16DType, &Bfloat16DType, &Bfloat16DType};
    PyArray_DTypeMeta* bb[2] = {&Bfloat16DType, &Bfloat16DType};
    PyArray_DTypeMeta* bbo[3] = {&Bfloat16DType, &Bfloat16DType, &PyArray_BoolDType};

    if (add_loop("add", bbb, 2, 1, (PyArrayMethod_StridedLoop*)&bf16_binary_loop<op_add>)) return -1;
    if (add_loop("subtract", bbb, 2, 1, (PyArrayMethod_StridedLoop*)&bf16_binary_loop<op_sub>))
        return -1;
    if (add_loop("multiply", bbb, 2, 1, (PyArrayMethod_StridedLoop*)&bf16_binary_loop<op_mul>))
        return -1;
    if (add_loop("true_divide", bbb, 2, 1, (PyArrayMethod_StridedLoop*)&bf16_binary_loop<op_div>))
        return -1;
    if (add_loop("negative", bb, 1, 1, (PyArrayMethod_StridedLoop*)&bf16_unary_loop<op_neg>))
        return -1;
    if (add_loop("absolute", bb, 1, 1, (PyArrayMethod_StridedLoop*)&bf16_unary_loop<op_abs>))
        return -1;
    if (add_loop("equal", bbo, 2, 1, (PyArrayMethod_StridedLoop*)&bf16_cmp_loop<cmp_eq>)) return -1;
    if (add_loop("not_equal", bbo, 2, 1, (PyArrayMethod_StridedLoop*)&bf16_cmp_loop<cmp_ne>))
        return -1;
    if (add_loop("less", bbo, 2, 1, (PyArrayMethod_StridedLoop*)&bf16_cmp_loop<cmp_lt>)) return -1;
    if (add_loop("less_equal", bbo, 2, 1, (PyArrayMethod_StridedLoop*)&bf16_cmp_loop<cmp_le>))
        return -1;
    if (add_loop("greater", bbo, 2, 1, (PyArrayMethod_StridedLoop*)&bf16_cmp_loop<cmp_gt>))
        return -1;
    if (add_loop("greater_equal", bbo, 2, 1, (PyArrayMethod_StridedLoop*)&bf16_cmp_loop<cmp_ge>))
        return -1;
    return 0;
}

// --------------------------------------------------------------------------
// Registration entry point (called from the nanobind module init).
// --------------------------------------------------------------------------
void register_bfloat16(nb::module_& m) {
    if (_import_array() < 0) throw std::runtime_error("numpy multiarray import failed");
    if (_import_umath() < 0) throw std::runtime_error("numpy umath import failed");

    // ---- scalar type ----
    PyBfloat16_Type = {PyVarObject_HEAD_INIT(nullptr, 0)};
    PyBfloat16_Type.tp_name = "universal_dtypes.bfloat16";
    PyBfloat16_Type.tp_basicsize = sizeof(PyBfloat16Object);
    PyBfloat16_Type.tp_flags = Py_TPFLAGS_DEFAULT;
    PyBfloat16_Type.tp_doc = "bfloat16 scalar (1 sign / 8 exponent / 7 mantissa)";
    PyBfloat16_Type.tp_new = PyBfloat16_New;
    PyBfloat16_Type.tp_repr = PyBfloat16_Repr;
    PyBfloat16_Type.tp_richcompare = PyBfloat16_RichCompare;
    bfloat16_as_number.nb_float = PyBfloat16_Float;
    PyBfloat16_Type.tp_as_number = &bfloat16_as_number;
    if (PyType_Ready(&PyBfloat16_Type) < 0)
        throw std::runtime_error("bfloat16 scalar type not ready");

    // ---- DTypeMeta: initialize the metaclass instance at runtime (C++-safe;
    // the C designated-initializer-after-HEAD idiom does not compile in C++) ----
    std::memset(&Bfloat16DType, 0, sizeof(Bfloat16DType));
    PyObject* dtobj = (PyObject*)&Bfloat16DType;
    Py_SET_REFCNT(dtobj, 1);
    Py_SET_TYPE(dtobj, &PyArrayDTypeMeta_Type);
    PyTypeObject* dt = (PyTypeObject*)&Bfloat16DType;
    dt->tp_name = "universal_dtypes.Bfloat16DType";
    dt->tp_base = &PyArrayDescr_Type;
    dt->tp_basicsize = sizeof(PyArray_Descr);
    dt->tp_flags = Py_TPFLAGS_DEFAULT;
    dt->tp_repr = bf16_descr_repr;
    dt->tp_str = bf16_descr_str;
    dt->tp_new = bf16_descr_new;
    if (PyType_Ready(dt) < 0) throw std::runtime_error("Bfloat16DType not ready");

    PyArrayDTypeMeta_Spec spec;
    spec.typeobj = &PyBfloat16_Type;
    spec.flags = NPY_DT_NUMERIC;
    spec.casts = build_casts();
    spec.slots = Bfloat16DType_Slots;
    spec.baseclass = nullptr;

    if (PyArrayInitDTypeMeta_FromSpec(&Bfloat16DType, &spec) < 0) {
        PyErr_Print();
        throw std::runtime_error("bfloat16 DType FromSpec failed");
    }

    if (Bfloat16DType.singleton == nullptr) {
        Bfloat16DType.singleton = PyArray_GetDefaultDescr(&Bfloat16DType);
    }
    if (Bfloat16DType.singleton == nullptr) {
        if (PyErr_Occurred()) PyErr_Print();
        throw std::runtime_error("bfloat16 default descriptor failed");
    }

    // free the heap-allocated cast specs (slot 0 is static)
    for (int i = 1; g_casts[i] != nullptr; i++) {
        free(g_casts[i]->dtypes);
        free(g_casts[i]->slots);
        free(g_casts[i]);
    }

    if (init_ufuncs() < 0) throw std::runtime_error("bfloat16 ufunc registration failed");

    // Expose the scalar type; np.dtype(ud.bfloat16) resolves via spec.typeobj.
    m.attr("bfloat16") = nb::borrow(reinterpret_cast<PyObject*>(&PyBfloat16_Type));
    m.attr("Bfloat16DType") = nb::borrow(reinterpret_cast<PyObject*>(&Bfloat16DType));
}
