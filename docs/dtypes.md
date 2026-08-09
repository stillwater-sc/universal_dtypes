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
ud.dtypes  # {"bfloat16": ..., "fp16": ..., "posit8": ..., "dd_cascade": ...}  every dtype
ud.posit_dtypes  # just the posit family
ud.cfloat_dtypes  # just the cfloat family (fp16, fp8e5m2)
ud.lns_dtypes  # just the lns family (lns16, lns32)
ud.fixpnt_dtypes  # fixed-point family (fixpnt16, fixpnt8)
ud.cascade_dtypes  # high-precision cascades (dd/td/qd_cascade)
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

## fixpnt (fixed-point)

Saturating fixed-point: a scaled integer with a fixed radix point, so resolution
is **uniform** across the range (unlike floating point's relative precision) and
**addition is exact within range** — the reason fixed-point is a DSP staple. There
is no NaN/Inf; out-of-range values clamp to ±maxpos (saturate) rather than wrap.

| config | `fixpnt<…>` | itemsize | format | range | resolution |
|--------|-------------|---------:|--------|-------|-----------|
| `fixpnt16` | `fixpnt<16,8,Saturate,uint16>` | 2 | Q8.8 | ±128 | 2⁻⁸ |
| `fixpnt8`  | `fixpnt<8,4,Saturate,uint8>`   | 1 | Q4.4 | ±8   | 2⁻⁴ |

## cascades (high precision)

Floating-point *expansions*: a value is an unevaluated sum of several IEEE
`double`s, giving many more bits of significand without arbitrary-precision
overhead. Arithmetic uses error-free transformations (two-sum / two-prod).
`ml_dtypes` has nothing like this.

| config | representation | itemsize | significand |
|--------|----------------|---------:|-------------|
| `dd_cascade` | double-double (2 × float64) | 16 | ~106 bits (~31 decimal digits) |
| `td_cascade` | triple-double (3 × float64) | 24 | ~159 bits (~47 decimal digits) |
| `qd_cascade` | quad-double (4 × float64) | 32 | ~212 bits (~62 decimal digits) |

These are the first **multi-word** dtypes (itemsize > one scalar word). Two
precision-aware rules apply:

- **Casts are lossy on the way out, exact on the way in:** `float64 → dd_cascade`
  is exact, but `dd_cascade → float64` is marked **unsafe** (it drops the low
  limbs). `astype` still performs it.
- **Comparisons and sort run at full precision** (via the type's own operators),
  so two values that differ below `float64` precision still order correctly —
  unlike a naive compare through `float64`.

Reductions (`sum`, `prod`) accumulate *in the expansion* — that extra precision is
the point. This is distinct from `mtl5`'s quire-based exact accumulation, which
stays in `mtl5` (see [`design.md`](design.md)).

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

## lns (logarithmic number system)

LNS stores a sign and the base-2 logarithm of the magnitude. Multiply and divide
become add/subtract of the stored logs (cheap, and exact in the exponent);
add/subtract are the hard operations and use Universal's Gaussian-log routines
(inexact, more so at low precision). There is no `ml_dtypes` counterpart.

| config | `lns<…>` | itemsize |
|--------|----------|---------:|
| `lns16` | `lns<16,8,uint16>`  | 2 |
| `lns32` | `lns<32,16,uint32>` | 4 |

`rbits` (the fractional resolution of the fixed-point exponent) is `8` for
`lns16` and `16` for `lns32` — Universal's canonical splits. LNS has dedicated
encodings for **zero** and **NaN** but **no infinity**, so `np.isinf` is always
`False` and `np.isfinite` is `not isnan`. Powers of two (and their products/
ratios) are exact; general add/sub are approximate — see the tests for the
tolerances (`lns32` is much tighter than `lns16`).

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
