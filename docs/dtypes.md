# Supported dtypes

`universal_dtypes` registers NumPy 2.x custom dtypes backed by Universal's C++
number types. Every dtype supports array creation, casts (to/from float/int/
bool), element-wise arithmetic, unary math ufuncs, comparisons, reductions, sort/
argsort, and pickling.

```python
import numpy as np, universal_dtypes as ud

a = np.array([1.0, 2.0, 3.0], dtype=ud.posit16)
np.sum(a * 2)  # arithmetic + reductions
a.astype(np.float32)  # casts
np.dtype("posit16")  # string-name resolution
```

## Discoverability

The set of dtypes is compiled in, so these registries are the source of truth for
what a given build ships:

```python
ud.dtypes  # {"bfloat16": ..., "fp16": ..., "posit8": ..., ...}  every dtype
ud.posit_dtypes  # just the posit family
ud.cfloat_dtypes  # just the cfloat family (fp16, fp8e5m2)
list(ud.dtypes)  # the names
np.dtype(ud.dtypes["posit12"])  # -> dtype(posit12)
```

## bfloat16

| name | bits | notes |
|------|------|-------|
| `bfloat16` | 16 | brain float (1 sign / 8 exp / 7 mantissa); IEEE NaN/Inf |

Bit-for-bit compatible with `ml_dtypes.bfloat16`. When `ml_dtypes` is also
installed it owns the `"bfloat16"` string name (we don't clobber it); pickling is
unaffected because it round-trips through the scalar type, not the name.

## cfloat (configurable float)

`cfloat` is Universal's IEEE-754-style configurable float (sign / exponent /
fraction, with optional subnormals / supernormals / saturation). The shipped
configs are chosen for exact parity with a reference:

| config | `cfloat<…>` | itemsize | equals |
|--------|-------------|---------:|--------|
| `fp16`    | `cfloat<16,5,uint16,true,false,false>` | 2 | `numpy.float16` (IEEE half) |
| `fp8e5m2` | `cfloat<8,5,uint8,true,false,false>`   | 1 | `ml_dtypes.float8_e5m2` |

Both are IEEE-style: they have `±inf`, `NaN`, and subnormals, so `np.isnan`,
`np.isinf`, and `np.isfinite` all work.

**`bfloat16`** is itself a `cfloat<16,8,…>` config, but it keeps its own dedicated
standalone implementation (for `ml_dtypes.bfloat16` parity) rather than going
through this family — see [`design.md`](design.md).

**`e4m3` is not shipped yet.** No Universal type is bit-exact with
`ml_dtypes.float8_e4m3fn` (OCP e4m3fn: max 448, no inf, overflow → NaN):
`cfloat<8,4>` is IEEE-style (has inf, max 240), and `microfloat` e4m3 saturates
large overflow to 448 instead of NaN. It's tracked as a follow-up.

## posit

Posits are tapered-precision reals with a single exceptional value **NaR** ("Not
a Real") — no ±Inf/NaN. `np.isnan` maps onto NaR; non-finite floats convert to
NaR; overflow saturates to maxpos/maxneg.

**Naming.** Bare `posit{nbits}` is the `es=2` standard (per the 2022 Posit
Standard). Other exponent sizes use `posit{nbits}e{es}`.

**Storage.** Each element occupies the smallest unsigned integer that holds
`nbits` bits (NumPy is byte-granular), so a non-power-of-two width is padded:

| config | `nbits` | `es` | itemsize |
|--------|--------:|-----:|---------:|
| `posit8`   |  8 | 2 | 1 |
| `posit16`  | 16 | 2 | 2 |
| `posit32`  | 32 | 2 | 4 |
| `posit64`  | 64 | 2 | 8 |
| `posit8e0` |  8 | 0 | 1 |
| `posit8e1` |  8 | 1 | 1 |
| `posit16e1`| 16 | 1 | 2 |
| `posit12`  | 12 | 2 | 2 |
| `posit20`  | 20 | 2 | 4 |
| `posit24`  | 24 | 2 | 4 |
| `posit28`  | 28 | 2 | 4 |
| `posit40`  | 40 | 2 | 8 |
| `posit48`  | 48 | 2 | 8 |

These configs use Universal's generic posit path (fast specializations exist only
for the standard `8/16/32/64, es=2`), so they are correct but not perf-tuned.

## Adding a posit configuration

The shipped set is curated to keep binary size and compile time in check; the
full `nbits × es` matrix is intentionally not instantiated. Adding a config is a
one-line entry in the codegen table in
[`python/src/posit.cpp`](../python/src/posit.cpp):

```cpp
#define UD_POSIT_LIST(X)          \
    X(8, 2, "posit8", Posit8)     \
    ...                            \
    X(10, 1, "posit10e1", Posit10e1)   // <- new config, then rebuild
```

`X(nbits, es, "scalar_name", ClassBase)` generates the traits struct and
registers the dtype; `es < nbits` is checked at compile time. Also export the new
name from [`universal_dtypes/__init__.py`](../universal_dtypes/__init__.py) and
add it to the `posit_dtypes` registry.

Because posits are C++ templates (compile-time), any config must be compiled in.
Choosing a config at Python runtime without a rebuild would require a parametric
DType — tracked as Stage 4 of issue #16.
