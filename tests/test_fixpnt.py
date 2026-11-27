"""fixpnt (fixed-point) NumPy dtype family — tests (issue #29).

Saturating fixed-point: fixpnt16 is Q8.8 (±128, resolution 2^-8), fixpnt8 is Q4.4
(±8, resolution 2^-4). Fixed-point has no NaN/Inf, uniform absolute resolution,
and exact addition within range.
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

FIXPNTS = [
    ("fixpnt16", ud.fixpnt16, 2, 128.0, 2.0**-8),
    ("fixpnt8", ud.fixpnt8, 1, 8.0, 2.0**-4),
]


def _params(name):
    return next(row for row in FIXPNTS if row[0] == name)


def test_dtype_resolves():
    for name, scalar, itemsize, _, _ in FIXPNTS:
        assert np.dtype(scalar).itemsize == itemsize
        assert np.dtype(name) == np.dtype(scalar)


def test_exact_roundtrip_on_grid():
    # values that land exactly on the fixed-point grid round-trip exactly
    for _, scalar, _, _, res in FIXPNTS:
        vals = [1.0, 2.0, -3.0, 0.5, res, 4 * res]
        a = np.array(vals, dtype=scalar)
        np.testing.assert_array_equal(a.astype(np.float64), vals)


def test_addition_is_exact_within_range():
    # fixed-point addition within range is exact (integer add of scaled values)
    for _, scalar, _, _, _ in FIXPNTS:
        a = np.array([0.5, 1.25, 0.0625], dtype=scalar)
        b = np.array([0.25, 2.0, 0.0625], dtype=scalar)
        np.testing.assert_array_equal((a + b).astype(np.float64), [0.75, 3.25, 0.125])


def test_no_nan_or_inf():
    for _, scalar, _, _, _ in FIXPNTS:
        a = np.array([1.0, -2.0, 0.0], dtype=scalar)
        assert not np.isnan(a).any()
        assert not np.isinf(a).any()
        assert np.isfinite(a).all()


def test_saturation():
    # out-of-range clamps to +-maxpos rather than wrapping (Saturate arithmetic)
    for _, scalar, _, maxabs, res in FIXPNTS:
        hi = float(np.array([1000.0], dtype=scalar)[0])
        lo = float(np.array([-1000.0], dtype=scalar)[0])
        assert maxabs - 1.0 <= hi <= maxabs  # near +maxpos
        assert -maxabs <= lo <= -(maxabs - 1.0)  # near -maxneg


def test_comparisons_sort_and_reductions():
    for _, scalar, _, _, _ in FIXPNTS:
        a = np.array([1.0, 2.0, 3.0], dtype=scalar)
        b = np.array([1.0, 5.0, 1.0], dtype=scalar)
        np.testing.assert_array_equal(a == b, [True, False, False])
        np.testing.assert_array_equal(a < b, [False, True, False])
        c = np.array([3.0, 1.0, 2.0], dtype=scalar)
        np.testing.assert_array_equal(np.sort(c).astype(np.float64), [1.0, 2.0, 3.0])
        # keep the sum within fixpnt8's Q4.4 range (+-8), not just fixpnt16's
        assert float(np.sum(np.array([1.0, 2.0, 0.5, 0.5], dtype=scalar))) == 4.0


def test_pickle_and_casts():
    import pickle

    for _, scalar, _, _, _ in FIXPNTS:
        a = np.array([1.0, 2.5, -3.0, 0.5], dtype=scalar)
        b = pickle.loads(pickle.dumps(a))
        assert b.dtype == np.dtype(scalar)
        np.testing.assert_array_equal(b.astype(np.float64), a.astype(np.float64))
        x = np.array([1, 2, 3], dtype=np.int64).astype(scalar)
        np.testing.assert_array_equal(x.astype(np.float64), [1.0, 2.0, 3.0])


def test_registry():
    # fixpnt_dtypes holds the general-purpose configs plus the DSP formats
    # (q7/q15/q31/iq24/q5_23 — covered by test_fixpnt_dsp.py)
    assert {"fixpnt16", "fixpnt8"} <= set(ud.fixpnt_dtypes)
    assert ud.fixpnt_dtypes["fixpnt16"] is ud.fixpnt16
    assert set(ud.fixpnt_dtypes) <= set(ud.dtypes)


# A division by zero saturates, by the dividend's sign, for every fixed-point
# dtype. Universal 5.15 and later throw fixpnt_divide_by_zero by default
# (universal#1907); a C++ exception cannot cross a NumPy ufunc loop, so it
# aborted the interpreter. python/CMakeLists.txt turns the throw off. The check
# runs in a child process, so a regression fails this test instead of killing
# the test run.
_DIVIDE_BY_ZERO = """
import sys, warnings
import numpy as np
import universal_dtypes as ud
warnings.simplefilter("ignore")
for name, scalar in ud.fixpnt_dtypes.items():
    a = np.array([1.0, -1.0, 0.25, -0.25], dtype=scalar)
    q = (a / np.zeros(4, dtype=scalar)).astype(np.float64)
    big = float(np.array([1e30], dtype=scalar).astype(np.float64)[0])
    small = float(np.array([-1e30], dtype=scalar).astype(np.float64)[0])
    expect = [big, small, big, small]
    if list(q) != expect:
        sys.exit(f"{name}: x / 0 gave {list(q)}, expected {expect}")
print("ok")
"""


def test_division_by_zero_saturates_instead_of_aborting():
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-c", _DIVIDE_BY_ZERO], capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0 and r.stdout.strip() == "ok", (
        f"exit {r.returncode}\nstdout: {r.stdout}\nstderr: {r.stderr[-2000:]}"
    )


def test_q31_sqrt_is_correctly_rounded_up_to_maxpos():
    """q31 sqrt rounds to nearest, and stays below 1 near maxpos.

    Under 2.x (Universal v5.11.0), the square root of maxpos (1 - 2^-31) wrapped
    to -1: 1581 of 5000 random operands gave a wrong result.
    """
    import math

    rng = np.random.default_rng(31)
    n = np.concatenate([[0, 1, 2, 2**31 - 2, 2**31 - 1], rng.integers(0, 2**31, 5000)])
    a = n.astype(np.int32).view(ud.q31)
    got = np.sqrt(a).view(np.int32).astype(np.int64)
    for k, r in zip(n.tolist(), got.tolist()):
        # sqrt(k / 2^31) * 2^31 = sqrt(k * 2^31), rounded to nearest, saturated
        m = math.isqrt(k << 31)
        if (2 * m + 1) ** 2 <= 4 * (k << 31):
            m += 1
        assert r == min(m, 2**31 - 1), (k, r, m)
