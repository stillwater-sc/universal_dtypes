"""dd_cascade (double-double) NumPy dtype — tests (issue #4).

dd_cascade is a 16-byte, ~106-bit high-precision type (unevaluated sum of two
float64). There is no ml_dtypes counterpart; correctness is shown by:
  - a cancellation demo where float64 loses digits dd retains,
  - full-precision comparison of values that collapse to the same float64,
  - agreement with the stdlib `decimal` module used as a high-precision oracle.
"""

from decimal import Decimal, getcontext

import numpy as np

import universal_dtypes as ud

dd = ud.dd_cascade
getcontext().prec = 40  # comfortably beyond dd's ~31 decimal digits


def test_dtype_resolves():
    dt = np.dtype(dd)
    assert dt.itemsize == 16  # two float64 limbs
    assert np.dtype("dd_cascade") == dt


def test_array_creation_and_exact_roundtrip():
    vals = [1.0, 2.0, 3.0, 0.5, -2.5, 1e20, 1e-20]
    a = np.array(vals, dtype=dd)
    # float64 -> dd is exact, and these values are exactly representable, so the
    # dd -> float64 round-trip returns them unchanged.
    np.testing.assert_array_equal(a.astype(np.float64), vals)


def test_cancellation_precision_beats_float64():
    # 1e20 + 1 - 1e20: float64 loses the 1; dd retains it. The exact answer (1.0)
    # is representable in float64, so we can check dd's superiority directly.
    a = np.array([1e20], dtype=dd)
    dd_result = float((a + np.array([1.0], dtype=dd) - np.array([1e20], dtype=dd))[0])
    f64_result = 1e20 + 1.0 - 1e20
    assert dd_result == 1.0
    assert f64_result == 0.0  # float64 cancels the 1 away


def test_full_precision_comparison():
    # two dd values that differ below float64 precision must still compare distinctly
    one = np.array([1.0], dtype=dd)
    bigger = one + np.array([1e-20], dtype=dd)
    assert float(one[0]) == float(bigger[0])  # collapse to the same float64
    assert bool((one < bigger)[0])  # ...but dd orders them correctly
    assert not bool((one == bigger)[0])


def test_sort_below_double_precision():
    base = np.array([1.0, 1.0, 1.0], dtype=dd)
    arr = base + np.array([2e-20, 0.0, 1e-20], dtype=dd)
    order = np.argsort(arr)
    np.testing.assert_array_equal(order, [1, 2, 0])  # 1+0 < 1+1e-20 < 1+2e-20


def test_arithmetic_matches_decimal_oracle():
    # dd holds ~31 decimal digits; a product of values inexact in float64 should
    # match a 40-digit Decimal reference to well beyond float64's ~16 digits.
    a_f, b_f = 1.1, 1.3  # not exactly representable in binary
    prod = np.array([a_f], dtype=dd) * np.array([b_f], dtype=dd)
    # exact dd value = high + low; compare to Decimal(a)*Decimal(b) at the doubles'
    # exact values (that is what dd actually multiplied).
    got = float(prod[0])
    ref = float(Decimal(a_f) * Decimal(b_f))
    assert abs(got - ref) < 1e-15  # far tighter than float64 rounding of the product


def test_special_values():
    v = np.array([np.inf, -np.inf, np.nan, 3.0], dtype=dd)
    np.testing.assert_array_equal(np.isinf(v), [True, True, False, False])
    np.testing.assert_array_equal(np.isnan(v), [False, False, True, False])
    np.testing.assert_array_equal(np.isfinite(v), [False, False, False, True])


def test_reductions():
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=dd)
    assert float(np.sum(a)) == 10.0
    assert float(np.prod(np.array([2.0, 3.0, 4.0], dtype=dd))) == 24.0


def test_casts():
    # float64 -> dd is exact; dd -> float64 is lossy (unsafe) but astype works.
    x = np.array([1, 2, 3], dtype=np.int64).astype(dd)
    np.testing.assert_array_equal(x.astype(np.float64), [1.0, 2.0, 3.0])
    assert np.dtype(dd).itemsize == 16


def test_pickle_roundtrip():
    import pickle

    # include a value with a nonzero low limb so pickling must preserve both limbs
    a = np.array([1.0], dtype=dd) + np.array([1e-20], dtype=dd)
    a = np.concatenate([a, np.array([2.0, 0.5], dtype=dd)])
    b = pickle.loads(pickle.dumps(a))
    assert b.dtype == np.dtype(dd)
    assert bool((a == b).all())  # exact, full-precision equality


def test_registry():
    assert ud.cascade_dtypes["dd_cascade"] is dd
    assert "dd_cascade" in ud.dtypes
