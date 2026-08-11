# Supported dtypes

`universal_dtypes` registers NumPy 2.x custom dtypes backed by Universal's C++
number types. Every dtype supports array creation, casts (to/from
`float16`/`float32`/`float64` **and between any two universal dtypes**),
element-wise arithmetic (including `**`, `minimum`/`maximum`, `clip`), unary math
ufuncs, comparisons, reductions (`sum`, `prod`, `min`, `max`), sort/argsort,
hashable scalars, and pickling.

```python
import numpy as np, universal_dtypes as ud

a = np.array([1.0, 2.0, 3.0], dtype=ud.posit16)
np.sum(a * ud.posit16(2))  # arithmetic + reductions
a.astype(np.float32)  # casts
np.dtype("posit16")  # string-name resolution
```

## Interoperating with builtin Python and NumPy types

**Python scalars work directly.** `int`, `float`, and `bool` operands are
absorbed into the array's dtype, on either side of the operator, for arithmetic,
comparisons, and `clip`:

```python
a * 2
a + 1.5
2 - a  # either side
a > 1  # comparisons too
np.clip(a, 0, 1)
np.where(a > 1, a, 0)  # np.result_type(a, 2) resolves as well
```

The scalar is **converted into the type first**, then the operation runs — so it
rounds or saturates by that type's own rules, identically to writing the
conversion out.

```python
a * 2 == a * ud.posit16(2)  # same result, always
```

That matters most for the bounded formats: `2` is not representable in `q15`
(range ±1), so it saturates to maxpos and `q15_arr * 2` multiplies by ~0.99997
rather than doubling. **A conversion that saturates reports it** — otherwise the
result looks plausible and the mistake is invisible:

```python
np.array([0.5], dtype=ud.q15) * 2
# RuntimeWarning: value 2 out of range for q15 (max 0.999969482421875); saturated
```

Ordinary rounding is *not* saturation and stays silent (`np.array(0.1,
dtype=ud.q15)` rounds, but 0.1 is well in range). Types that carry an infinity
(`cfloat`, `bfloat16`) overflow to `inf` rather than clamping, so they never
report.

Silencing it depends on which conversion path you are on, because the two use
different channels:

```python
# array -> array (astype, array operands): NumPy's float error state
with np.errstate(over="ignore"):  # or "raise"
    np.array([2.0]).astype(ud.q15)

# scalar -> element (a * 2, np.array(x, dtype=T), np.full): the warnings module
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    a * 2
```

The scalar path cannot use `np.errstate`: NumPy clears the float status before
running a ufunc loop, and the scalar operand is converted before that clear, so
the status would be wiped before anything reported it. Both default to a visible
`RuntimeWarning`.

**A concrete NumPy operand still raises**, by design — rounding a whole `float64`
array into a low-precision type is a data-loss decision that should be explicit:

```python
a * np.float64(2)  # UFuncTypeError
a * np.array([2.0])  # UFuncTypeError — use a.astype(np.float64) or an .astype() cast
a * 2j  # UFuncTypeError — absorbing complex would drop the imaginary part
```

**Array creation works through the usual entry points**, `np.arange` included:

```python
np.arange(8, dtype=ud.posit16)
np.arange(0, 1, 0.25, dtype="posit16")
np.zeros(4, dtype=ud.posit16)  # zeros / ones / full / empty
np.linspace(0, 1, 8).astype(ud.posit16)  # linspace has no dtype= for these
```

`arange` computes each element as `start + i*delta` from the absolute index and
rounds once, rather than accumulating `v[i-1] + delta` — the same rule NumPy
uses for its own floats, and it avoids compounding a rounding error at every
step. `delta` is taken from the first two (already rounded) elements, so for a
step that is not exactly representable, `np.arange(0, 1, 0.1, dtype=ud.posit16)`
differs slightly from `np.arange(0, 1, 0.1).astype(ud.posit16)`. NumPy's own
`float16` differs from its `float64` counterpart in exactly the same way.

Two consequences worth knowing: on the bounded formats the progression
**saturates** rather than raising (`np.arange(3, dtype=ud.q15)` is
`[0, maxpos, maxpos]`, since `q15` holds only ±1), and for the `dd`/`td`/`qd`
cascades the progression is computed in `double`, so it carries double precision
rather than the type's full significand.

(Scalar promotion was [#55](https://github.com/stillwater-sc/universal_dtypes/issues/55),
`arange` [#56](https://github.com/stillwater-sc/universal_dtypes/issues/56).)

## Casting between dtypes

`astype` works between **any two** universal dtypes, not just to/from the builtin
NumPy types:

```python
a = np.array([1.5, 2.25], dtype=ud.posit16)
a.astype(ud.posit32)  # posit -> posit
a.astype(ud.bfloat16)  # across number systems
a.astype(ud.dd_cascade)  # into a high-precision cascade
```

Casts convert in the **value domain** (the represented real number), not by
reinterpreting bits. All such casts are classified **unsafe** (a different number
system may round), so `astype` performs them but implicit promotion does not.

Of the **builtin** NumPy types, only `float16`/`float32`/`float64` cast in both
directions. There is no outbound integer or boolean cast, and the inbound ones
are incomplete, so route integer and boolean data through `float64`:

```python
a.astype(np.float64).astype(np.int32)  # posit16 -> int
np.array([1, 2], dtype=np.int32).astype(np.float64).astype(ud.posit16)  # int -> posit16
```

The conversion goes through a compensated multi-term expansion built from each
type's own arithmetic, so it preserves the **source's full precision** — even when
that exceeds `float64`'s 53 bits (`posit64` and the `dd`/`td`/`qd` cascades). A
plain `float64` intermediate would silently drop those low bits; casting
`posit64 → qd_cascade → posit64`, for instance, is bit-exact.

## Reductions and the accumulation contract

Reductions accumulate **in the type itself**, in array order — there is no hidden
wider accumulator. This is deliberate: the accumulation precision is the caller's
choice, and exact/quire accumulation is [`mtl5`](https://github.com/stillwater-sc/mtl5-python)'s
job, not this package's.

| operation | behavior |
|-----------|----------|
| `sum`, `prod` | naive in-type accumulation, in order. `sum([]) → 0`, `prod([]) → 1` (identities, rounded into the type — see note). |
| `min`, `max` | in-type, compared at **full precision** (the cascades order below `float64`). `min([])`/`max([])` raise, as in NumPy. |
| `mean` | **not** computed in-type — see below. |
| math ufuncs (`exp`, `sqrt`, `**`, …) | computed in `double`, then rounded back into the type. |

The empty-reduction identity is the type's own representation of `0`/`1`: the
fractional DSP formats (`q7`/`q15`/`q31`, range ±1) can't hold `1.0`, so their
`prod([])` is the nearest representable value rather than exactly `1`.

**In-type accumulation swamps.** Because there is no wider accumulator, small
addends are lost against a large running sum:

```python
a = np.array([100.0] + [0.01] * 50, dtype=ud.posit16)
np.sum(a)  # 100.0  — the 0.01s vanish in posit16
np.sum(a.astype(np.float64))  # ~100.5 — accumulate wider by casting first
```

**Wider accumulation and `mean`: cast first.** A wider accumulation dtype is not
selected implicitly, so `np.sum(a, dtype=np.float64)` and `np.mean(a)` raise
`TypeError` (there is no in-type divide-by-count for `mean`, by design). Accumulate
in the precision you want by casting the array:

```python
a.astype(np.float64).sum()  # wider accumulation
a.astype(np.float64).mean()  # the supported mean
```

## Persistence and byte order

Arrays round-trip through the usual mechanisms — **on the same platform**:

```python
pickle.loads(pickle.dumps(a))  # pickle
np.load(path, allow_pickle=True)  # np.save / np.load (allow_pickle on load)
np.frombuffer(a.tobytes(), dtype=a.dtype)  # raw bytes
```

`np.save`/`np.load` needs `allow_pickle=True` on load, because NumPy stores a
custom dtype via the pickle protocol rather than the plain `.npy` binary header.

**These dtypes are native-endian only.** Array storage is the raw element bytes,
with no byte-order tag: `np.dtype(ud.posit16).byteorder` is `'|'` (not
applicable), and NumPy 2.x's new-style DType API does not support `newbyteorder`
or `byteswap` for them — don't call those (they are unsupported, and on some
NumPy 2.x versions they crash rather than raise). Consequently a file written on
one platform is **not** guaranteed to load on a platform of different endianness. The target
ecosystem is little-endian, so this is an accepted limitation for now;
cross-endian support is purely additive and can be added later without breaking
anyone. (Individual *scalars* pickle by value — their `__reduce__` stores the
`double` — so a single scalar is portable; arrays store raw bytes.)

## pandas integration (optional)

Install the extra and import the (opt-in) integration module — the core package
never imports pandas:

```bash
pip install 'universal_dtypes[pandas]'
```

```python
import pandas as pd
import universal_dtypes.pandas_ext  # registers the pandas dtypes

s = pd.Series([1.5, 2.25, 3.0], dtype="posit16")  # by name
s.max()
s.astype("posit32")
s.astype(float)  # reductions + casts
pd.DataFrame({"x": pd.array([1.0, 2.0], dtype="bfloat16")})
```

Every universal dtype gets a pandas `ExtensionDtype`/`ExtensionArray` pair, thinly
backed by the NumPy dtype (pure Python, no MTL5). The classes are also exposed by
CamelCase name — `universal_dtypes.pandas_ext.Posit16Dtype` / `Posit16Array` — for
downstream re-export. `astype` between universal dtypes reuses the value-domain
cross-casts above; `mean`/`std` follow the same "cast first" rule as the NumPy
reductions.

## Discoverability

The set of dtypes is compiled in, so these registries are the source of truth for
what a given build ships:

```python
ud.dtypes  # {"bfloat16": ..., "fp16": ..., "posit8": ..., "dd_cascade": ...}  every dtype
ud.posit_dtypes  # just the posit family
ud.takum_dtypes  # just the takum family
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

### DSP processor formats (TI / Analog Devices)

The fixed-point formats native to commercial DSPs are also registered (all
saturating). The fractional `q7`/`q15`/`q31` are shared across TI, ADI, and ARM
CMSIS-DSP; `iq24` and `q5_23` are vendor-specific:

| config | `fixpnt<…>` | itemsize | format | range | used by |
|--------|-------------|---------:|--------|-------|---------|
| `q7`    | `fixpnt<8,7,Saturate,uint8>`    | 1 | Q1.7  | ±1   | TI/ADI/ARM 8-bit |
| `q15`   | `fixpnt<16,15,Saturate,uint16>` | 2 | Q1.15 | ±1   | TI C5000, ADI ADSP-21xx / Blackfin |
| `q31`   | `fixpnt<32,31,Saturate,uint32>` | 4 | Q1.31 | ±1   | TI C6000, ADI Blackfin / SHARC |
| `iq24`  | `fixpnt<32,24,Saturate,uint32>` | 4 | Q8.24 | ±128 | TI C2000 IQmath (IQ24 default) |
| `q5_23` | `fixpnt<28,23,Saturate,uint32>` | 4 | 5.23  | ±16  | ADI SigmaDSP audio |

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

`cfloat` is Universal's configurable float: sign / exponent / fraction with
optional subnormals, supernormals and saturation, parameterized consistently from
a couple of bits up to thousands. The shipped configs are two useful points in
that space:

| config | `cfloat<…>` | itemsize | layout |
|--------|-------------|---------:|--------|
| `fp16`    | `cfloat<16,5,uint16,true,false,false>` | 2 | 1 sign / 5 exponent / 10 fraction |
| `fp8e5m2` | `cfloat<8,5,uint8,true,false,false>`   | 1 | 1 sign / 5 exponent / 2 fraction |

Both have `±inf`, `NaN` and subnormals, so `np.isnan`, `np.isinf` and
`np.isfinite` all work, and both round to nearest-even.

### Not interchangeable with IEEE binary16 or ml_dtypes

**`fp16` is not a drop-in for `numpy.float16`, and `fp8e5m2` is not a drop-in for
`ml_dtypes.float8_e5m2`.** They share a field layout and produce the same values
for finite data, but the encodings are *not* the same type and this package does
not offer compatibility with either as a guarantee.

`cfloat` applies one encoding rule uniformly across its whole parameter range.
IEEE-754 does not: its special-value conventions are specific to the sizes the
standard happens to enumerate, and do not generalize to arbitrary
`(nbits, es)`. Where the two disagree, `cfloat` is the definition this package
implements — the Stillwater KPU honors the `cfloat` encoding, so matching IEEE
bit-for-bit is not a goal we would trade `cfloat` consistency for.

Concretely, the disagreement is confined to the `±inf` patterns — 4 encodings out
of 65536 for `fp16`, 4 of 256 for `fp8e5m2`:

| pattern | `fp16` decodes as | IEEE binary16 decodes as |
|---------|-------------------|--------------------------|
| `0x7c00` | `NaN` | `+inf` |
| `0x7ffe` | `+inf` | `NaN` |
| `0xfc00` | `NaN` | `-inf` |
| `0xfffe` | `-inf` | `NaN` |

`fp8e5m2` diverges the same way at `0x7c`/`0x7e`/`0xfc`/`0xfe`. Every other
all-ones-exponent pattern is `NaN` under both.

What this means in practice:

- **`astype` is correct in both directions** — `inf` stays `inf`, `NaN` stays
  `NaN`, finite values are unchanged. Converting is the supported path and always
  has been.
- **Do not reinterpret a raw buffer** across `fp16` and `numpy.float16`
  (`np.frombuffer(x.tobytes(), ...)`). Zero-copy aliasing is **not supported**
  and will not be: those four patterns change meaning. Finite-only data happens
  to survive it, which makes the trap quiet — use `astype`.

A recoding shim, so a copy across the boundary can fix up `±inf` explicitly
rather than relying on the accident that finite data aliases cleanly, is a
plausible future addition.

**`bfloat16`** is its own dedicated standalone implementation going through
single precision float hardware rather than going through the cfloat<> family — see
[`design.md`](design.md).

**`e4m3` is not shipped.** `cfloat<8,4>` is available in Universal if a config in
that shape is wanted; note it is a `cfloat`, with `cfloat` semantics (it has
`inf`, max 240), and is not the OCP e4m3fn format.

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

## takum

The **linear takum** (Hunhold, 2024, [arXiv:2404.18603](https://arxiv.org/abs/2404.18603)),
a tapered-precision format like posit but with a very different range/precision
split. Universal's logarithmic variant is a separate upstream type
(`takum_log`), so the bare `takum{nbits}` names here are permanently the linear
encoding.

Value is `(-1)^sign * (1 + f) * 2^c`, with the magnitude laid out as
`[S:1][D:1][R:rbits][C:r bits][M:p bits]`. Storage is two's complement: zero is
all-zero bits, **NaR** is the sign bit alone, and comparing the raw signed
integer equals comparing the real value. Like posit there is a single NaR and
**no infinity**, so `np.isinf` is always `False` and any non-finite float
converts to NaR.

| config | `takum<…>` | itemsize | maxpos |
|--------|-----------|---------:|-------:|
| `takum8`  | `takum<8,3,uint8>`   | 1 | 8.8e71 |
| `takum16` | `takum<16,3,uint16>` | 2 | 5.6e76 |
| `takum32` | `takum<32,3,uint32>` | 4 | 5.8e76 |
| `takum64` | `takum<64,3,uint64>` | 8 | 5.8e76 |

`rbits = 3` is the takum spec's regime width and Universal's default.

**Why both takum and posit.** takum's dynamic range is essentially *independent
of width* — flat at ~5.8e76 from 16 bits up — so wider configs buy precision
rather than range. posit does the opposite:

| width | posit maxpos | takum maxpos |
|-------|-------------:|-------------:|
| 8  | 1.7e7  | 8.8e71 |
| 16 | 7.2e16 | 5.6e76 |
| 32 | 1.3e36 | 5.8e76 |
| 64 | 4.5e74 | 5.8e76 |

So `takum8` reaches magnitudes `posit8` cannot express at all, at some cost in
precision near ±1, while by 64 bits the two ranges converge.

### takum64 arithmetic is capped at double precision

Universal implements takum's `+ - * /` by converting both operands to `double`,
computing there, and converting back. For `takum8`/`takum16`/`takum32` that is
**exact** — their significands are far below double's 53 bits, so it is a single
correct rounding, the same rule the [math ufuncs](#reductions-and-the-accumulation-contract)
already follow.

`takum64`'s significand reaches ~59 bits, so its operands are rounded to `double`
*before* the operation and any detail below that is lost:

```python
a = np.array([1.0], dtype=ud.takum64)
# a value one encoding step above 1.0 is distinct, and sorts correctly...
# ...but adding zero to it does not round-trip.
```

What still carries the full width: the **encoding**, **casts**, **comparisons**,
**sort**, and `min`/`max`. Only arithmetic is capped. The out-cast to `float64`
is graded *unsafe* for `takum64` accordingly (it is safe for the narrower
configs), and cross-dtype casts round-trip exactly for anything `double` can
hold, but not below it — the cross-cast expansion builds its residual terms with
the type's own arithmetic, so it inherits the same cap.

`posit64` does **not** share this: Universal gives it native arithmetic, so it
round-trips and computes at its full width. If you need more than double's
precision from arithmetic, prefer `posit64` or the `dd`/`td`/`qd` cascades.

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
