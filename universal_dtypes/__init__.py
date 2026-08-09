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
- ``fixpnt16`` / ``fixpnt8`` — saturating fixed-point (Q8.8 / Q4.4): uniform
  resolution and exact addition within range (DSP).
- ``q7`` / ``q15`` / ``q31`` / ``iq24`` / ``q5_23`` — the standard fixed-point
  formats of TI and Analog Devices DSPs (Q1.7/Q1.15/Q1.31 fractional, TI IQmath
  Q8.24, ADI SigmaDSP 5.23).
- ``dd_cascade`` / ``td_cascade`` / ``qd_cascade`` — double-double (~106-bit,
  16-byte), triple-double (~159-bit, 24-byte), and quad-double (~212-bit, 32-byte)
  high-precision types; arithmetic uses error-free transformations.

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
    Fixpnt8DType,
    Fixpnt16DType,
    Fp8e5m2DType,
    Fp16DType,
    Iq24DType,
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
    Q5_23DType,
    Q7DType,
    Q15DType,
    Q31DType,
    QdCascadeDType,
    TdCascadeDType,
    bfloat16,
    build_info,
    dd_cascade,
    fixpnt8,
    fixpnt16,
    fp8e5m2,
    fp16,
    iq24,
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
    q5_23,
    q7,
    q15,
    q31,
    qd_cascade,
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

fixpnt_dtypes = {
    "fixpnt16": fixpnt16,
    "fixpnt8": fixpnt8,
    # standard DSP fixed-point formats (TI / Analog Devices)
    "q7": q7,  # Q1.7  — TI/ADI/ARM 8-bit fractional
    "q15": q15,  # Q1.15 — TI C5000, ADI 21xx/Blackfin
    "q31": q31,  # Q1.31 — TI C6000, ADI Blackfin/SHARC
    "iq24": iq24,  # Q8.24 — TI C2000 IQmath (IQ24)
    "q5_23": q5_23,  # 5.23  — ADI SigmaDSP audio
}

# high-precision cascades (multi-word storage): the complete dd/td/qd family.
cascade_dtypes = {
    "dd_cascade": dd_cascade,
    "td_cascade": td_cascade,
    "qd_cascade": qd_cascade,
}

dtypes = {
    "bfloat16": bfloat16,
    **cfloat_dtypes,
    **posit_dtypes,
    **lns_dtypes,
    **fixpnt_dtypes,
    **cascade_dtypes,
}

__all__ = [
    "__version__",
    "bfloat16",
    "Bfloat16DType",
    "build_info",
    "dtypes",
    "cfloat_dtypes",
    "lns_dtypes",
    "fixpnt_dtypes",
    "cascade_dtypes",
    "posit_dtypes",
    "posit16_roundtrip",
    # fixpnt scalar types + dtype classes
    "fixpnt16",
    "fixpnt8",
    "Fixpnt16DType",
    "Fixpnt8DType",
    # DSP fixed-point formats (TI / ADI)
    "q7",
    "q15",
    "q31",
    "iq24",
    "q5_23",
    "Q7DType",
    "Q15DType",
    "Q31DType",
    "Iq24DType",
    "Q5_23DType",
    # cascade (high-precision) types
    "dd_cascade",
    "DdCascadeDType",
    "td_cascade",
    "TdCascadeDType",
    "qd_cascade",
    "QdCascadeDType",
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
