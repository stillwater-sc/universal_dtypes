# universal_dtypes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/universal-dtypes.svg)](https://pypi.org/project/universal-dtypes/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![CI](https://github.com/stillwater-sc/universal_dtypes/actions/workflows/ci.yml/badge.svg)](https://github.com/stillwater-sc/universal_dtypes/actions/workflows/ci.yml)
[![Wheels](https://github.com/stillwater-sc/universal_dtypes/actions/workflows/wheels.yml/badge.svg)](https://github.com/stillwater-sc/universal_dtypes/actions/workflows/wheels.yml)

> **Stable — v2.0.0.** The public API is committed to Semantic Versioning
> guarantees. (`2.0.0` is the *first* stable major: `1.0.0` was published in
> error very early and yanked, and PyPI reserves a version number permanently —
> see [`docs/RELEASING.md`](docs/RELEASING.md) §3.)

`universal_dtypes` is a stand-alone Python package that registers the Stillwater
[Universal](https://github.com/stillwater-sc/universal) number systems as
first-class [NumPy](https://numpy.org) dtypes, so alternative real-number formats
can be used directly in NumPy (and pandas) arrays. It is modeled on
[`ml_dtypes`](https://github.com/jax-ml/ml_dtypes), which does the same for the
machine-learning float formats used by JAX and TensorFlow — and goes further:
posits, fixed-point (including the commercial DSP Q-formats), the logarithmic
number system, and double/triple/quad-double cascades have no `ml_dtypes`
counterpart.

Linear algebra over these types (dense/sparse solvers, and the **quire**-based
exact accumulation policy) lives in the sister package
[mtl5-python](https://github.com/stillwater-sc/mtl5-python), which consumes these
dtypes. See [Relationship to MTL5](#relationship-to-mtl5).

## The dtypes

28 dtypes ship, across six families. Full tables — bit layouts, ranges,
resolutions, and the exact Universal template configuration behind each — are in
[`docs/dtypes.md`](docs/dtypes.md).

| family | dtypes | notes |
|--------|--------|-------|
| **bfloat16** | `bfloat16` | brain float (1/8/7); bit-for-bit `ml_dtypes.bfloat16` |
| **posit** | `posit8`, `posit16`, `posit32`, `posit64`; `posit8e0`, `posit8e1`, `posit16e1`; `posit12`, `posit20`, `posit24`, `posit28`, `posit40`, `posit48` | tapered precision, single **NaR**. Bare `posit{n}` is `es=2`; `posit{n}e{es}` selects the exponent size |
| **cfloat** | `fp16`, `fp8e5m2` | configurable IEEE-style floats; round identically to `numpy.float16` and `ml_dtypes.float8_e5m2` |
| **lns** | `lns16`, `lns32` | logarithmic number system (`lns<16,8>` / `lns<32,16>`) |
| **fixpnt** | `fixpnt8`, `fixpnt16`; `q7`, `q15`, `q31`, `iq24`, `q5_23` | saturating fixed-point, incl. the standard TI / Analog Devices / ARM DSP formats |
| **cascade** | `dd_cascade`, `td_cascade`, `qd_cascade` | double/triple/quad-double expansions: ~106 / ~159 / ~212 significand bits |

The set is compiled in, so the registries are the source of truth for what a
given build ships:

```python
import universal_dtypes as ud

list(ud.dtypes)  # every name
ud.posit_dtypes  # just one family (also cfloat_/lns_/fixpnt_/cascade_dtypes)
ud.build_info()  # {'version': '2.0.0', 'universal': True, 'dtypes': True}
```

Adding a posit configuration is a one-line table entry plus a rebuild — see
[`docs/dtypes.md`](docs/dtypes.md#adding-a-posit-configuration).

## Installation

```bash
pip install universal_dtypes
```

Binary wheels are published for CPython 3.10–3.12 on Linux x86-64
(manylinux_2_28), macOS arm64, and Windows x64. Other platforms and Python
versions build from the sdist, which needs a **C++20 compiler** and CMake (the
build fetches the Universal headers via `FetchContent`).

**NumPy 2.x is required** at both build and run time: these are NEP-42 dtypes
(`PyArrayInitDTypeMeta_FromSpec`, ArrayMethod casts, ufunc loops), an API that
does not exist in NumPy 1.x.

Optional extras:

```bash
pip install 'universal_dtypes[pandas]'   # pandas ExtensionDtype/ExtensionArray
pip install 'universal_dtypes[dev]'      # pytest, ml_dtypes, pandas
```

From source:

```bash
git clone https://github.com/stillwater-sc/universal_dtypes
cd universal_dtypes
pip install -e ".[dev]"
pytest
```

## Example usage

```python
import numpy as np
import universal_dtypes as ud

a = np.array([1.0, 2.0, 3.0], dtype=ud.posit16)
b = a * a                    # element-wise arithmetic, in posit16
c = np.sum(a)                # reductions (in-type accumulation — see below)
n = np.sqrt(a)               # math ufuncs
s = np.sort(a)               # sort / argsort
m = a > ud.posit16(1)        # comparisons
```

Every dtype supports array creation, casts, element-wise arithmetic and math
ufuncs (including `**`, `minimum`/`maximum`, `clip`), comparisons, reductions,
sort/argsort, hashable scalars, and pickling.

The dtypes also register under their string names:

```python
np.dtype("posit16")
np.zeros(4, dtype="posit16")
np.full(4, 2.5, dtype="fp8e5m2")
```

**Mixing with Python scalars is not implicit.** There is no promotion loop for
`int`/`float`, so `a * 2` raises; use a typed scalar or a 0-d array instead:

```python
a * ud.posit16(2)                     # ok
a * np.array(2.0, dtype=ud.posit16)   # ok
a * 2                                 # UFuncTypeError — no loop for (posit16, int)
```

Likewise `np.arange` does not accept these dtypes; build in `float64` and cast
(`zeros`/`ones`/`full`/`empty` do work):

```python
np.linspace(0, 1, 8).astype(ud.posit16)
```

Both gaps are tracked —
[#55](https://github.com/stillwater-sc/universal_dtypes/issues/55) and
[#56](https://github.com/stillwater-sc/universal_dtypes/issues/56) — and closing
them is backward-compatible, so neither is frozen by the v2 API.

Worked, runnable problems — one per number-system family, plus cross-family
application studies in math, ML, control, and DSP — live in
[`examples/`](examples/) (see its [README](examples/README.md)). Each one asserts
its result, so they double as regression tests.

## Casting between dtypes

`astype` works between **any two** universal dtypes, not just to and from the
builtin NumPy types:

```python
a = np.array([1.5, 2.25], dtype=ud.posit16)
a.astype(ud.posit32)     # posit -> posit
a.astype(ud.bfloat16)    # across number systems
a.astype(ud.dd_cascade)  # into a high-precision cascade
a.astype(np.float32)     # and float16 / float32 / float64
```

Of the builtin NumPy types, only `float16`/`float32`/`float64` cast in both
directions. Integer and boolean arrays go through `float64`:

```python
a.astype(np.float64).astype(np.int32)                          # posit16 -> int
np.array([1, 2], dtype=np.int32).astype(np.float64).astype(ud.posit16)
```

Casts convert in the **value domain** (the represented real number), not by
reinterpreting bits, and go through a compensated multi-term expansion built from
each type's own arithmetic — so they preserve the source's full precision even
when it exceeds `float64`'s 53 bits (`posit64`, and the cascades). All such casts
are classified *unsafe* (a different number system may round), so `astype`
performs them but implicit promotion does not.

## Reductions and the accumulation contract

Reductions accumulate **in the type itself**, in array order — there is no hidden
wider accumulator. This is deliberate: the accumulation precision is the caller's
choice. Consequently `np.sum(a, dtype=np.float64)` and `np.mean(a)` raise; cast
first to pick your precision.

```python
a.astype(np.float64).sum()   # wider accumulation
a.astype(np.float64).mean()  # the supported mean
```

The full contract — empty-reduction identities, full-precision `min`/`max`, and
the `double`-then-round rule for math ufuncs — is pinned in
[`docs/dtypes.md`](docs/dtypes.md#reductions-and-the-accumulation-contract).

Arrays round-trip through pickle, `np.save`/`np.load` (with `allow_pickle=True`),
and raw bytes, but the dtypes are **native-endian only** — see the
[persistence and byte-order contract](docs/dtypes.md#persistence-and-byte-order).

## pandas integration (optional)

An opt-in module gives every universal dtype a pandas
`ExtensionDtype`/`ExtensionArray` pair. The core package never imports pandas.

```python
import pandas as pd
import universal_dtypes.pandas_ext  # registers the pandas dtypes

s = pd.Series([1.5, 2.25, 3.0], dtype="posit16")
s.max()
s.astype("posit32")
pd.DataFrame({"x": pd.array([1.0, 2.0], dtype="bfloat16")})
```

## Number-format background

### Posits — `posit8`, `posit16`, `posit32`, `posit64`, …

A posit `posit<nbits, es>` encodes a real number with a sign bit, a run-length
*regime* field, up to `es` exponent bits, and the remaining bits as fraction.
Precision is **tapered**: highest for magnitudes near ±1, decreasing toward `0`
and the extremes. Posits have **no subnormals** and **no separate infinities or
NaNs** — a single exceptional value **NaR** ("Not a Real") represents overflow,
division by zero, and other undefined results; overflow saturates to
maxpos/maxneg. `np.isnan` maps onto NaR.

### Configurable floats — `fp16`, `fp8e5m2`

Universal's `cfloat` is a parameterized IEEE-754-style float (sign / exponent /
fraction). Both shipped configs are IEEE-style with `±inf`, `NaN`, and
subnormals, chosen to round exactly like a reference implementation. The parity
covers every finite encoding, but Universal places `±inf` at a different bit
pattern than IEEE (4 patterns out of the whole space), so a raw buffer
containing infinities cannot be reinterpreted across the two —
[#57](https://github.com/stillwater-sc/universal_dtypes/issues/57). `astype` is
correct either way. (`e4m3` is not shipped yet; see
[`docs/dtypes.md`](docs/dtypes.md#cfloat-configurable-float).)

### Fixed-point — `fixpnt8/16`, `q7`, `q15`, `q31`, `iq24`, `q5_23`

Fixed-point `fixpnt<nbits, rbits>` is a two's-complement integer interpreted as a
value scaled by `2^-rbits` — a fixed absolute resolution across the whole range,
and **exact addition within range**, which is why it is a DSP staple. All shipped
configs saturate on overflow rather than wrapping; there are no non-finite
values. `q7`/`q15`/`q31` are the fractional formats shared by TI, ADI, and ARM
CMSIS-DSP; `iq24` is TI C2000 IQmath and `q5_23` is ADI SigmaDSP audio.

### Logarithmic number system — `lns16`, `lns32`

LNS stores a sign and the base-2 logarithm of the magnitude. Multiplication and
division become addition and subtraction of the stored logarithms — cheap, and
exact for powers of two — at the cost of harder addition and subtraction, which
use Universal's Gaussian-log routines. LNS has encodings for zero and NaN but
**no infinity**.

### Cascades — `dd_cascade`, `td_cascade`, `qd_cascade`

Floating-point *expansions*: a value is an unevaluated sum of 2, 3, or 4 IEEE
`double`s, giving ~106 / ~159 / ~212 significand bits without
arbitrary-precision overhead. Arithmetic uses error-free transformations
(two-sum / two-prod); comparisons and sort run at full precision.

## Quirks of low-precision arithmetic

Because there is no wider accumulator, naive accumulation loses information —
summing in `posit16` rounds after **every** partial sum, and small addends are
swamped by a large running total:

```python
import numpy as np
import universal_dtypes as ud

x = np.array([100.0] + [0.01] * 50, dtype=ud.posit16)
np.sum(x)                     # 100.0  — every 0.01 vanishes against the total
np.sum(x.astype(np.float64))  # ≈100.5 — accumulate wider by casting first
```

The standard fix is to accumulate in higher precision (cast first, as above).
Universal's posits also provide the **quire** — a fixed-point accumulator wide
enough to hold an exact sum of products, rounding only once at the end. The quire
is an *algorithm* concern (how you accumulate), so it lives in the linear-algebra
layer, not in the dtype: it is exposed by
[mtl5-python](https://github.com/stillwater-sc/mtl5-python) through
`accumulator=` on `dot`, `norm`, and the other reduction-bearing operations.
`universal_dtypes` deliberately gives ordinary NumPy semantics; when you need an
exact reduction, reach for MTL5:

```python
import mtl5

# exact posit16 sum-of-products via the quire (single final rounding)
mtl5.dot(x, np.ones_like(x), accumulator="quire")
```

The two answers will differ on ill-conditioned data — by design.

## Relationship to MTL5

`universal_dtypes` is the **element type** and its scalar / element-wise
arithmetic. [mtl5-python](https://github.com/stillwater-sc/mtl5-python) is what
you *do* with arrays of them: dense/sparse linear algebra, and the quire /
accumulator mixed-precision policy. mtl5-python depends on `universal_dtypes` for
the dtypes; `universal_dtypes` does **not** depend on MTL5. See
[`docs/design.md`](docs/design.md) for the full rationale and the package
boundary.

## Documentation

- [`docs/dtypes.md`](docs/dtypes.md) — the complete dtype reference: per-family
  tables, casts, the reduction contract, persistence, pandas, discoverability.
- [`docs/design.md`](docs/design.md) — architecture, the package boundary, and
  the design decisions behind the API.
- [`docs/RELEASING.md`](docs/RELEASING.md) — release process and versioning
  policy.
- [`CHANGELOG.md`](CHANGELOG.md) — per-release history.
- [`examples/`](examples/) — runnable worked problems and application studies.

## License

`universal_dtypes` is licensed under the MIT License — see [LICENSE](LICENSE).
Copyright © 2026 Stillwater Supercomputing, Inc.

It builds on [Universal](https://github.com/stillwater-sc/universal) (MIT), whose
header-only number-system implementations provide the arithmetic behind each
dtype.
