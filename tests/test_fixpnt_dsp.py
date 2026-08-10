"""DSP fixed-point formats (TI / Analog Devices) — tests.

q7/q15/q31 are the Q1.(n-1) fractional formats common to TI (C5000/C6000), ADI
(ADSP-21xx/Blackfin/SHARC) and ARM CMSIS-DSP; iq24 is TI C2000 IQmath's Q8.24;
q5_23 is ADI SigmaDSP's 5.23 audio format. All saturating.
"""

import numpy as np
import pytest

import universal_dtypes as ud

# Saturation is exercised deliberately here, which now reports — see #60 and
# test_saturation_warning.py, which asserts that signal. These tests are about
# the resulting values.
pytestmark = pytest.mark.filterwarnings(
    "ignore:(value .* out of range|overflow encountered in cast)"
)

# name, scalar, itemsize, approx max magnitude (range endpoint)
DSP = [
    ("q7", ud.q7, 1, 1.0),
    ("q15", ud.q15, 2, 1.0),
    ("q31", ud.q31, 4, 1.0),
    ("iq24", ud.iq24, 4, 128.0),
    ("q5_23", ud.q5_23, 4, 16.0),
]


def test_dtype_resolves():
    for name, scalar, itemsize, _ in DSP:
        assert np.dtype(scalar).itemsize == itemsize
        assert np.dtype(name) == np.dtype(scalar)


def test_exact_small_fractions():
    # small negative powers of two are exact in every one of these formats
    vals = [0.5, -0.25, 0.125, -0.0625]
    for _, scalar, _, _ in DSP:
        a = np.array(vals, dtype=scalar)
        np.testing.assert_array_equal(a.astype(np.float64), vals)


def test_saturation_matches_range():
    # out-of-range clamps to the format's max magnitude (saturating DSP arithmetic)
    for name, scalar, _, maxmag in DSP:
        hi = float(np.array([1e6], dtype=scalar)[0])
        lo = float(np.array([-1e6], dtype=scalar)[0])
        # within 1.0 of the range endpoint (fractional formats top out just below 1)
        assert maxmag - 1.0 <= hi <= maxmag
        assert -maxmag <= lo <= -(maxmag - 1.0)


def test_arithmetic_within_range():
    for _, scalar, _, _ in DSP:
        a = np.array([0.25, 0.5], dtype=scalar)
        b = np.array([0.125, 0.25], dtype=scalar)
        np.testing.assert_array_equal((a + b).astype(np.float64), [0.375, 0.75])
        np.testing.assert_array_equal((a - b).astype(np.float64), [0.125, 0.25])
        # 0.5 * 0.5 = 0.25 is representable in all of them
        np.testing.assert_array_equal(
            (np.array([0.5], scalar) * np.array([0.5], scalar)).astype(np.float64), [0.25]
        )


def test_precision_ladder():
    # more fractional bits -> finer resolution near 0.  q7 < q15 < q31; and q31's
    # 31 fractional bits beat iq24's 24 near small values.
    x = 0.1  # not exactly representable in any of them
    err = {name: abs(float(np.array([x], scalar)[0]) - x) for name, scalar, _, _ in DSP}
    assert err["q15"] < err["q7"]
    assert err["q31"] < err["q15"]
    assert err["q31"] < err["iq24"]


def test_pickle_and_registry():
    import pickle

    for _, scalar, _, _ in DSP:
        a = np.array([0.5, -0.25, 0.125], dtype=scalar)
        b = pickle.loads(pickle.dumps(a))
        assert b.dtype == np.dtype(scalar)
        np.testing.assert_array_equal(b.astype(np.float64), a.astype(np.float64))
    for name, scalar, _, _ in DSP:
        assert ud.fixpnt_dtypes[name] is scalar
        assert name in ud.dtypes
