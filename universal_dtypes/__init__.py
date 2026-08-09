"""NumPy dtypes for the Stillwater Universal number systems.

Provides NumPy 2.x custom dtypes backed by Universal's C++ number types:

- ``bfloat16`` — brain floating point.
- ``posit8`` / ``posit16`` / ``posit32`` / ``posit64`` — the standard posit sizes
  (``es=2``), plus selected other configurations (``posit8e0``/``posit8e1``/
  ``posit16e1`` and the non-power-of-two widths ``posit12``/``20``/``24``/``28``/
  ``40``/``48``). The ``posit{nbits}e{es}`` name form selects the exponent size;
  bare ``posit{nbits}`` is ``es=2``.
- ``fp16`` (IEEE half) and ``fp8e5m2`` — configurable floats (``cfloat``),
  bit-compatible with ``numpy.float16`` and ``ml_dtypes.float8_e5m2``.
- ``lns16`` / ``lns32`` — logarithmic number system (``lns<16,8>`` / ``lns<32,16>``):
  multiply/divide are exact in the exponent; add/subtract use Universal's
  Gaussian-log routines.
- ``dd_cascade`` / ``td_cascade`` — double-double (~106-bit, 16-byte) and
  triple-double (~159-bit, 24-byte) high-precision types; arithmetic uses
  error-free transformations.

Each dtype supports array creation, casts, element-wise arithmetic and math
ufuncs, comparisons, reductions, sort, and pickling. Use them like any dtype::

    import numpy as np, universal_dtypes as ud
    a = np.array([1.0, 2.0, 3.0], dtype=ud.posit16)
    np.sum(a * 2)

See ``docs/design.md`` for the architecture and ``python/src/universal_dtype.hpp``
for the reusable registration harness.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Single source of truth: installed package metadata, falling back to the version
# compiled into the extension when running from a source tree without metadata.
try:
    __version__ = _pkg_version("universal_dtypes")
except _PackageNotFoundError:
    try:
        from universal_dtypes._core import __version__  # noqa: F811
    except ImportError:
        __version__ = "0.0.0-dev"

from universal_dtypes._core import (  # noqa: E402
    Bfloat16DType,
    DdCascadeDType,
    Fp8e5m2DType,
    Fp16DType,
    Lns16DType,
    Lns32DType,
    Posit8DType,
    Posit8e0DType,
    Posit8e1DType,
    Posit12DType,
    Posit16DType,
    Posit16e1DType,
    Posit20DType,
    Posit24DType,
    Posit28DType,
    Posit32DType,
    Posit40DType,
    Posit48DType,
    Posit64DType,
    TdCascadeDType,
    bfloat16,
    build_info,
    dd_cascade,
    fp8e5m2,
    fp16,
    lns16,
    lns32,
    posit8,
    posit8e0,
    posit8e1,
    posit12,
    posit16,
    posit16_roundtrip,
    posit16e1,
    posit20,
    posit24,
    posit28,
    posit32,
    posit40,
    posit48,
    posit64,
    td_cascade,
)

# Discoverability: name -> scalar type for every registered dtype. `np.dtype()`
# of any value here yields the NumPy dtype; iterate the mapping to enumerate what
# this build ships. (The shipped set is compiled in — see python/src/posit.cpp's
# UD_POSIT_LIST — so these registries are the source of truth for "what exists".)
posit_dtypes = {
    "posit8": posit8,
    "posit16": posit16,
    "posit32": posit32,
    "posit64": posit64,
    "posit8e0": posit8e0,
    "posit8e1": posit8e1,
    "posit16e1": posit16e1,
    "posit12": posit12,
    "posit20": posit20,
    "posit24": posit24,
    "posit28": posit28,
    "posit40": posit40,
    "posit48": posit48,
}

cfloat_dtypes = {
    "fp16": fp16,
    "fp8e5m2": fp8e5m2,
}

lns_dtypes = {
    "lns16": lns16,
    "lns32": lns32,
}

# high-precision cascades (multi-word storage); qd_cascade arrives with #6.
cascade_dtypes = {
    "dd_cascade": dd_cascade,
    "td_cascade": td_cascade,
}

dtypes = {"bfloat16": bfloat16, **cfloat_dtypes, **posit_dtypes, **lns_dtypes, **cascade_dtypes}

__all__ = [
    "__version__",
    "bfloat16",
    "Bfloat16DType",
    "build_info",
    "dtypes",
    "cfloat_dtypes",
    "lns_dtypes",
    "cascade_dtypes",
    "posit_dtypes",
    "posit16_roundtrip",
    # cascade (high-precision) types
    "dd_cascade",
    "DdCascadeDType",
    "td_cascade",
    "TdCascadeDType",
    # cfloat scalar types + dtype classes
    "fp16",
    "fp8e5m2",
    "Fp16DType",
    "Fp8e5m2DType",
    # lns scalar types + dtype classes
    "lns16",
    "lns32",
    "Lns16DType",
    "Lns32DType",
    # posit scalar types
    "posit8",
    "posit16",
    "posit32",
    "posit64",
    "posit8e0",
    "posit8e1",
    "posit16e1",
    "posit12",
    "posit20",
    "posit24",
    "posit28",
    "posit40",
    "posit48",
    # posit dtype classes
    "Posit8DType",
    "Posit16DType",
    "Posit32DType",
    "Posit64DType",
    "Posit8e0DType",
    "Posit8e1DType",
    "Posit16e1DType",
    "Posit12DType",
    "Posit20DType",
    "Posit24DType",
    "Posit28DType",
    "Posit40DType",
    "Posit48DType",
]
