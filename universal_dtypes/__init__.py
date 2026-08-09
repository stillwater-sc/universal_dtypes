"""NumPy dtypes for the Stillwater Universal number systems.

Provides NumPy 2.x custom dtypes backed by Universal's C++ number types:

- ``bfloat16`` — brain floating point.
- ``posit8`` / ``posit16`` / ``posit32`` / ``posit64`` — the standard posit sizes
  (``es=2``), plus selected other configurations (``posit8e0``/``posit8e1``/
  ``posit16e1`` and the non-power-of-two widths ``posit12``/``20``/``24``/``28``/
  ``40``/``48``). The ``posit{nbits}e{es}`` name form selects the exponent size;
  bare ``posit{nbits}`` is ``es=2``.

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
    bfloat16,
    build_info,
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
)

__all__ = [
    "__version__",
    "bfloat16",
    "Bfloat16DType",
    "build_info",
    "posit16_roundtrip",
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
