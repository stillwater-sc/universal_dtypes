# universal_dtypes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/universal-dtypes.svg)](https://pypi.org/project/universal-dtypes/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)

<!-- Unittests / Wheel-build badges will be added alongside the CI workflows. -->

> **Status — early development.** The repository is scaffolded and the design is
> settled (see [`docs/design.md`](docs/design.md)); the NumPy dtype
> implementation has not started. The installation and usage snippets below
> describe the **intended** API — `universal_dtypes` is **not yet published to
> PyPI**.

`universal_dtypes` is a stand-alone Python package that registers the Stillwater
[Universal](https://github.com/stillwater-sc/universal) number systems as
first-class [NumPy](https://numpy.org) dtypes, so alternative real-number formats
can be used directly in NumPy (and pandas) arrays. It is modeled on
[`ml_dtypes`](https://github.com/jax-ml/ml_dtypes), which does the same for
machine-learning float formats used by JAX and TensorFlow.

The following dtypes are planned:

- `posit8`, `posit16`, `posit32`, `posit64` — tapered-precision posits
  (Posit Standard, `es = 2`)
- `fp8`, `fp16` — configurable low-precision floats (Universal `cfloat`)
- `fixpnt8`, `fixpnt16` — fixed-point
- `lns16`, `lns32` — logarithmic number system

Linear algebra over these types (dense/sparse solvers, and the **quire**-based
exact accumulation policy) lives in the sister package
[mtl5-python](https://github.com/stillwater-sc/mtl5-python), which consumes these
dtypes. See [Relationship to MTL5](#relationship-to-mtl5).

## Installation

`universal_dtypes` will be published to PyPI as
[`universal-dtypes`](https://pypi.org/project/universal-dtypes/) (PEP 503
normalizes the underscore to a hyphen; `pip install universal_dtypes` also
resolves):

```bash
pip install universal_dtypes
```

To run the tests:

```bash
pip install universal_dtypes[dev]
pytest
```

To build from source (the build fetches the Universal C++ headers via CMake
FetchContent; a C++20 compiler is required):

```bash
git clone https://github.com/stillwater-sc/universal_dtypes
cd universal_dtypes
pip install -e ".[dev]"
```

## Example usage

```python
import numpy as np
import universal_dtypes

a = np.array([1.0, 2.0, 3.0], dtype=universal_dtypes.posit16)
b = a * 2  # element-wise arithmetic, in posit16
c = np.sum(a)  # NumPy reductions (naive posit accumulation)
n = np.sqrt(a)  # NumPy ufuncs
```

The dtypes also register under their string names, so NumPy's usual APIs accept
them:

```python
np.dtype("posit16")
np.zeros(4, dtype="posit16")
np.arange(8, dtype="fp8")
```

## Specifications of implemented number formats

### Posits — `posit8`, `posit16`, `posit32`, `posit64`

A posit `posit<nbits, es>` encodes a real number with a sign bit, a run-length
*regime* field, up to `es` exponent bits, and the remaining bits as fraction.
Precision is **tapered**: it is highest for magnitudes near ±1 and decreases
toward `0` and `±∞`. Posits have **no subnormals** and **no separate infinities
or NaNs** — a single exceptional value **NaR** ("Not a Real") represents
overflow, division by zero, and other undefined results. The standard sizes use
`es = 2` (`posit8 = posit<8,2>`, `posit16 = posit<16,2>`, and so on).

### Configurable floats — `fp8`, `fp16`

Universal's `cfloat` is a parameterized IEEE-754-style float (sign / exponent /
fraction). `fp8` and `fp16` are low-precision configurations for mixed-precision
work. Unlike posits, they follow the familiar exponent/mantissa layout and can
carry subnormals, infinities, and NaN depending on the configuration.

### Fixed-point — `fixpnt8`, `fixpnt16`

Fixed-point `fixpnt<nbits, rbits>` is a two's-complement integer interpreted as a
value scaled by `2^-rbits` — a fixed absolute resolution across its whole range
(in contrast to floating formats), with configurable rounding and
saturation/wrap-around on overflow.

### Logarithmic number system — `lns16`, `lns32`

The logarithmic number system stores a sign and the base-2 logarithm of the
magnitude. Multiplication and division become addition and subtraction of the
stored logarithms (cheap and exact in the exponent), at the cost of more
expensive addition and subtraction.

## Encodings and exceptional values

Each type is a true NumPy dtype with a fixed item size — `posit8`/`fixpnt8`/`fp8`
occupy one byte, `posit16`/`fixpnt16`/`fp16`/`lns16` two bytes, `posit32`/`lns32`
four, and `posit64` eight. Exceptional values differ by family and are **not**
interchangeable with IEEE semantics:

- **posits** have a single **NaR** and no ±inf/NaN; overflow saturates toward the
  largest-magnitude finite posit rather than producing infinity.
- **cfloat** (`fp8`/`fp16`) may expose ±inf and NaN per its configuration.
- **fixpnt** has no non-finite values; out-of-range results saturate or wrap.
- **lns** carries its own encodings for zero and its exceptional value.

Casts to and from `float32`/`float64` (and the integer types) round to the target
format; converting a non-finite IEEE value into a posit yields NaR.

## Quirks of low-precision arithmetic

As with any low-precision format, naive accumulation loses information. Summing
many values in `posit16` rounds after **every** partial sum:

```python
import numpy as np
import universal_dtypes

x = np.ones(4096, dtype=universal_dtypes.posit16)
np.sum(x)  # rounds each partial sum in posit16 — not 4096 exactly
```

The standard fix is to accumulate in higher precision. Universal's posits also
provide the **quire** — a fixed-point accumulator wide enough to hold an exact
sum of products, rounding only once at the end. The quire is an *algorithm*
concern (how you accumulate), so it lives in the linear-algebra layer, not in the
dtype: it is exposed by [mtl5-python](https://github.com/stillwater-sc/mtl5-python)
through `accumulator=` on `dot`, `norm`, and the other reduction-bearing
operations. `universal_dtypes` deliberately gives ordinary NumPy semantics for
the dtype; when you need an exact reduction, reach for MTL5:

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

## License

`universal_dtypes` is licensed under the MIT License — see [LICENSE](LICENSE).
Copyright © 2026 Stillwater Supercomputing, Inc.

It builds on [Universal](https://github.com/stillwater-sc/universal) (MIT), whose
header-only number-system implementations provide the arithmetic behind each
dtype.
