// bfloat16 NumPy dtype — MVP for issue #3 (legacy NumPy user-dtype C-API).
//
// Built against NumPy 1.x (non-opaque PyArray_Descr); the wheel runs on 2.x via
// NumPy's forward-ABI guarantee. bf16<->float32 use canonical round-to-nearest-
// even, which matches ml_dtypes.bfloat16; routing the round through Universal's
// bfloat16 is a follow-up (numerically identical). Arithmetic computes in
// float32 and rounds to bf16 — the correct bf16 semantics.
//
// Modeled on NumPy's own user-dtype example (numpy _rational_tests.c).

#include <Python.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#define PY_ARRAY_UNIQUE_SYMBOL universal_dtypes_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL universal_dtypes_UFUNC_API
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#include <nanobind/nanobind.h>
namespace nb = nanobind;

// --------------------------------------------------------------------------
// bf16 <-> float32 (canonical round-to-nearest-even)
// --------------------------------------------------------------------------
static inline float bf16_bits_to_float(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static inline uint16_t float_to_bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if (std::isnan(f)) {
        return static_cast<uint16_t>((u >> 16) | 0x0040u);  // quiet NaN
    }
    // round-to-nearest-even
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t rounding_bias = 0x00007FFFu + lsb;
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

// --------------------------------------------------------------------------
// Python scalar type: universal_dtypes.bfloat16
// --------------------------------------------------------------------------
typedef struct {
    PyObject_HEAD
    uint16_t bits;
} PyBfloat16Object;

static PyTypeObject PyBfloat16_Type;

static inline int is_bfloat16(PyObject* obj) {
    return PyObject_TypeCheck(obj, &PyBfloat16_Type);
}

static PyObject* bfloat16_from_bits(uint16_t bits) {
    PyBfloat16Object* o = PyObject_New(PyBfloat16Object, &PyBfloat16_Type);
    if (o) o->bits = bits;
    return reinterpret_cast<PyObject*>(o);
}

// Convert an arbitrary Python object to bf16 bits. Returns 0 on success.
static int bfloat16_bits_from_pyobject(PyObject* obj, uint16_t* out) {
    if (is_bfloat16(obj)) {
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
    return PyUnicode_FromFormat("%R", PyFloat_FromDouble(static_cast<double>(f)));
}

static PyObject* PyBfloat16_RichCompare(PyObject* a, PyObject* b, int op) {
    double x, y;
    if (is_bfloat16(a)) x = bf16_bits_to_float(reinterpret_cast<PyBfloat16Object*>(a)->bits);
    else { x = PyFloat_AsDouble(a); if (x == -1.0 && PyErr_Occurred()) Py_RETURN_NOTIMPLEMENTED; }
    if (is_bfloat16(b)) y = bf16_bits_to_float(reinterpret_cast<PyBfloat16Object*>(b)->bits);
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
// dtype registration
// --------------------------------------------------------------------------
static int npy_bfloat16 = -1;  // assigned type number
static PyArray_ArrFuncs bfloat16_arrfuncs;
static PyArray_Descr bfloat16_descr;

static PyObject* bf16_getitem(void* data, void* /*arr*/) {
    uint16_t bits;
    std::memcpy(&bits, data, sizeof(bits));
    return bfloat16_from_bits(bits);
}

static int bf16_setitem(PyObject* item, void* data, void* /*arr*/) {
    uint16_t bits;
    if (bfloat16_bits_from_pyobject(item, &bits) < 0) return -1;
    std::memcpy(data, &bits, sizeof(bits));
    return 0;
}

static void bf16_copyswap(void* dst, void* src, int swap, void* /*arr*/) {
    if (src) std::memcpy(dst, src, sizeof(uint16_t));
    if (swap) {
        char* p = static_cast<char*>(dst);
        std::swap(p[0], p[1]);
    }
}

static void bf16_copyswapn(void* dst, npy_intp dstride, void* src, npy_intp sstride,
                           npy_intp n, int swap, void* /*arr*/) {
    char* d = static_cast<char*>(dst);
    char* s = static_cast<char*>(src);
    for (npy_intp i = 0; i < n; i++) {
        if (s) std::memcpy(d, s, sizeof(uint16_t));
        if (swap) std::swap(d[0], d[1]);
        d += dstride;
        if (s) s += sstride;
    }
}

static int bf16_compare(const void* a, const void* b, void* /*arr*/) {
    uint16_t ba, bb;
    std::memcpy(&ba, a, 2);
    std::memcpy(&bb, b, 2);
    float fa = bf16_bits_to_float(ba), fb = bf16_bits_to_float(bb);
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

static npy_bool bf16_nonzero(void* data, void* /*arr*/) {
    uint16_t bits;
    std::memcpy(&bits, data, 2);
    return bf16_bits_to_float(bits) != 0.0f ? NPY_TRUE : NPY_FALSE;
}

// casts: bf16 -> T
template <typename T>
static void cast_from_bf16(void* from, void* to, npy_intp n, void* /*fa*/, void* /*ta*/) {
    const uint16_t* in = static_cast<const uint16_t*>(from);
    T* out = static_cast<T*>(to);
    for (npy_intp i = 0; i < n; i++) out[i] = static_cast<T>(bf16_bits_to_float(in[i]));
}
// casts: T -> bf16
template <typename T>
static void cast_to_bf16(void* from, void* to, npy_intp n, void* /*fa*/, void* /*ta*/) {
    const T* in = static_cast<const T*>(from);
    uint16_t* out = static_cast<uint16_t*>(to);
    for (npy_intp i = 0; i < n; i++) out[i] = float_to_bf16_bits(static_cast<float>(in[i]));
}

// binary ufunc loops (compute in float32, round to bf16)
template <float (*Op)(float, float)>
static void bf16_binary_loop(char** args, npy_intp const* dims, npy_intp const* steps, void*) {
    char *i0 = args[0], *i1 = args[1], *o = args[2];
    npy_intp n = dims[0];
    for (npy_intp k = 0; k < n; k++) {
        uint16_t a, b;
        std::memcpy(&a, i0, 2);
        std::memcpy(&b, i1, 2);
        uint16_t r = float_to_bf16_bits(Op(bf16_bits_to_float(a), bf16_bits_to_float(b)));
        std::memcpy(o, &r, 2);
        i0 += steps[0]; i1 += steps[1]; o += steps[2];
    }
}
static float op_add(float a, float b) { return a + b; }
static float op_sub(float a, float b) { return a - b; }
static float op_mul(float a, float b) { return a * b; }
static float op_div(float a, float b) { return a / b; }

template <float (*Op)(float)>
static void bf16_unary_loop(char** args, npy_intp const* dims, npy_intp const* steps, void*) {
    char *i0 = args[0], *o = args[1];
    npy_intp n = dims[0];
    for (npy_intp k = 0; k < n; k++) {
        uint16_t a;
        std::memcpy(&a, i0, 2);
        uint16_t r = float_to_bf16_bits(Op(bf16_bits_to_float(a)));
        std::memcpy(o, &r, 2);
        i0 += steps[0]; o += steps[1];
    }
}
static float op_neg(float a) { return -a; }
static float op_abs(float a) { return std::fabs(a); }

// comparison loops (bf16, bf16 -> bool)
template <bool (*Cmp)(float, float)>
static void bf16_cmp_loop(char** args, npy_intp const* dims, npy_intp const* steps, void*) {
    char *i0 = args[0], *i1 = args[1], *o = args[2];
    npy_intp n = dims[0];
    for (npy_intp k = 0; k < n; k++) {
        uint16_t a, b;
        std::memcpy(&a, i0, 2);
        std::memcpy(&b, i1, 2);
        npy_bool r = Cmp(bf16_bits_to_float(a), bf16_bits_to_float(b)) ? NPY_TRUE : NPY_FALSE;
        *reinterpret_cast<npy_bool*>(o) = r;
        i0 += steps[0]; i1 += steps[1]; o += steps[2];
    }
}
static bool cmp_eq(float a, float b) { return a == b; }
static bool cmp_ne(float a, float b) { return a != b; }
static bool cmp_lt(float a, float b) { return a < b; }
static bool cmp_le(float a, float b) { return a <= b; }
static bool cmp_gt(float a, float b) { return a > b; }
static bool cmp_ge(float a, float b) { return a >= b; }

static int register_ufunc_binary(PyObject* umath, const char* name,
                                 PyUFuncGenericFunction fn) {
    PyObject* ufunc = PyObject_GetAttrString(umath, name);
    if (!ufunc) return -1;
    int types[3] = {npy_bfloat16, npy_bfloat16, npy_bfloat16};
    int r = PyUFunc_RegisterLoopForType(reinterpret_cast<PyUFuncObject*>(ufunc), npy_bfloat16,
                                        fn, types, nullptr);
    Py_DECREF(ufunc);
    return r;
}
static int register_ufunc_unary(PyObject* umath, const char* name,
                                PyUFuncGenericFunction fn) {
    PyObject* ufunc = PyObject_GetAttrString(umath, name);
    if (!ufunc) return -1;
    int types[2] = {npy_bfloat16, npy_bfloat16};
    int r = PyUFunc_RegisterLoopForType(reinterpret_cast<PyUFuncObject*>(ufunc), npy_bfloat16,
                                        fn, types, nullptr);
    Py_DECREF(ufunc);
    return r;
}
static int register_ufunc_cmp(PyObject* umath, const char* name,
                              PyUFuncGenericFunction fn) {
    PyObject* ufunc = PyObject_GetAttrString(umath, name);
    if (!ufunc) return -1;
    int types[3] = {npy_bfloat16, npy_bfloat16, NPY_BOOL};
    int r = PyUFunc_RegisterLoopForType(reinterpret_cast<PyUFuncObject*>(ufunc), npy_bfloat16,
                                        fn, types, nullptr);
    Py_DECREF(ufunc);
    return r;
}

void register_bfloat16(nb::module_& m) {
    if (_import_array() < 0) throw std::runtime_error("numpy.core.multiarray import failed");
    if (_import_umath() < 0) throw std::runtime_error("numpy.core.umath import failed");

    // scalar type
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
    if (PyType_Ready(&PyBfloat16_Type) < 0) throw std::runtime_error("bfloat16 scalar type not ready");

    // arrfuncs
    PyArray_InitArrFuncs(&bfloat16_arrfuncs);
    bfloat16_arrfuncs.getitem = bf16_getitem;
    bfloat16_arrfuncs.setitem = bf16_setitem;
    bfloat16_arrfuncs.copyswap = bf16_copyswap;
    bfloat16_arrfuncs.copyswapn = bf16_copyswapn;
    bfloat16_arrfuncs.compare = bf16_compare;
    bfloat16_arrfuncs.nonzero = bf16_nonzero;

    // descr
    PyObject_INIT(reinterpret_cast<PyObject*>(&bfloat16_descr), &PyArrayDescr_Type);
    bfloat16_descr.typeobj = &PyBfloat16_Type;
    bfloat16_descr.kind = 'V';
    bfloat16_descr.type = 'E';
    bfloat16_descr.byteorder = '=';
    bfloat16_descr.flags = NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM;
    bfloat16_descr.elsize = sizeof(uint16_t);
    bfloat16_descr.alignment = alignof(uint16_t);
    bfloat16_descr.f = &bfloat16_arrfuncs;

    npy_bfloat16 = PyArray_RegisterDataType(&bfloat16_descr);
    if (npy_bfloat16 < 0) throw std::runtime_error("bfloat16 dtype registration failed");
    Py_INCREF(&PyBfloat16_Type);
    PyDict_SetItemString(PyBfloat16_Type.tp_dict, "dtype",
                         reinterpret_cast<PyObject*>(&bfloat16_descr));

    // casts bf16 <-> float32/float64/int64
    PyArray_RegisterCastFunc(&bfloat16_descr, NPY_FLOAT, cast_from_bf16<float>);
    PyArray_RegisterCastFunc(&bfloat16_descr, NPY_DOUBLE, cast_from_bf16<double>);
    PyArray_RegisterCastFunc(&bfloat16_descr, NPY_LONGLONG, cast_from_bf16<long long>);
    PyArray_RegisterCanCast(&bfloat16_descr, NPY_FLOAT, NPY_NOSCALAR);
    PyArray_RegisterCanCast(&bfloat16_descr, NPY_DOUBLE, NPY_NOSCALAR);

    auto reg_to = [](int from, PyArray_VectorUnaryFunc* fn) {
        PyArray_Descr* d = PyArray_DescrFromType(from);
        PyArray_RegisterCastFunc(d, npy_bfloat16, fn);
        Py_DECREF(d);
    };
    reg_to(NPY_FLOAT, cast_to_bf16<float>);
    reg_to(NPY_DOUBLE, cast_to_bf16<double>);
    reg_to(NPY_LONG, cast_to_bf16<long>);
    reg_to(NPY_LONGLONG, cast_to_bf16<long long>);
    reg_to(NPY_BOOL, cast_to_bf16<npy_bool>);

    // ufunc loops
    PyObject* umath = PyImport_ImportModule("numpy.core.umath");
    if (!umath) throw std::runtime_error("numpy.core.umath import failed");
    register_ufunc_binary(umath, "add", bf16_binary_loop<op_add>);
    register_ufunc_binary(umath, "subtract", bf16_binary_loop<op_sub>);
    register_ufunc_binary(umath, "multiply", bf16_binary_loop<op_mul>);
    register_ufunc_binary(umath, "true_divide", bf16_binary_loop<op_div>);
    register_ufunc_unary(umath, "negative", bf16_unary_loop<op_neg>);
    register_ufunc_unary(umath, "absolute", bf16_unary_loop<op_abs>);
    register_ufunc_cmp(umath, "equal", bf16_cmp_loop<cmp_eq>);
    register_ufunc_cmp(umath, "not_equal", bf16_cmp_loop<cmp_ne>);
    register_ufunc_cmp(umath, "less", bf16_cmp_loop<cmp_lt>);
    register_ufunc_cmp(umath, "less_equal", bf16_cmp_loop<cmp_le>);
    register_ufunc_cmp(umath, "greater", bf16_cmp_loop<cmp_gt>);
    register_ufunc_cmp(umath, "greater_equal", bf16_cmp_loop<cmp_ge>);
    Py_DECREF(umath);

    // expose the scalar type; np.dtype(ud.bfloat16) resolves via typeobj
    m.attr("bfloat16") = nb::borrow(reinterpret_cast<PyObject*>(&PyBfloat16_Type));
}
