"""td_cascade (triple-double) NumPy dtype — tests (issue #5).

td_cascade is a 24-byte, ~159-bit high-precision type (unevaluated sum of three
float64) — the precision tier above dd_cascade. Same multi-word machinery; the
distinguishing test is a three-magnitude cancellation that needs more than
double-double precision to retain.
"""

from decimal import Decimal, getcontext

import numpy as np

import universal_dtypes as ud

td = ud.td_cascade
getcontext().prec = 60  # beyond td's ~47 decimal digits


def test_dtype_resolves():
    dt = np.dtype(td)
    assert dt.itemsize == 24  # three float64 limbs
    assert np.dtype("td_cascade") == dt


def test_array_creation_and_exact_roundtrip():
    vals = [1.0, 2.0, 3.0, 0.5, -2.5, 1e20, 1e-20]
    a = np.array(vals, dtype=td)
    np.testing.assert_array_equal(a.astype(np.float64), vals)


def test_deep_cancellation_beyond_double_double():
    # 1e20 + 1 + 1e-20 - 1e20 - 1 == 1e-20 exactly. Spanning three magnitudes
    # (1e20, 1, 1e-20) needs ~159-bit intermediate precision — double-double is
    # not enough; triple-double retains it. The exact answer is float64-representable.
    parts = [1e20, 1.0, 1e-20, -1e20, -1.0]
    acc = np.array([parts[0]], dtype=td)
    for p in parts[1:]:
        acc = acc + np.array([p], dtype=td)
    assert float(acc[0]) == 1e-20
    # float64 loses the small terms inside the 1e20 sums, ending grossly wrong
    # (at -1.0 here) — nowhere near the true 1e-20.
    f64 = 0.0
    for p in parts:
        f64 += p
    assert f64 == -1.0
    assert abs(f64 - 1e-20) > 0.5


def test_full_precision_comparison():
    one = np.array([1.0], dtype=td)
    bigger = one + np.array([1e-30], dtype=td)  # below double AND double-double range
    assert float(one[0]) == float(bigger[0])
    assert bool((one < bigger)[0])
    assert not bool((one == bigger)[0])


def test_sort_below_double_precision():
    base = np.array([1.0, 1.0, 1.0], dtype=td)
    arr = base + np.array([2e-30, 0.0, 1e-30], dtype=td)
    np.testing.assert_array_equal(np.argsort(arr), [1, 2, 0])


def test_arithmetic_matches_decimal_oracle():
    a_f, b_f = 1.1, 1.3
    prod = np.array([a_f], dtype=td) * np.array([b_f], dtype=td)
    got = float(prod[0])
    ref = float(Decimal(a_f) * Decimal(b_f))
    assert abs(got - ref) < 1e-15


def test_special_values():
    v = np.array([np.inf, -np.inf, np.nan, 3.0], dtype=td)
    np.testing.assert_array_equal(np.isinf(v), [True, True, False, False])
    np.testing.assert_array_equal(np.isnan(v), [False, False, True, False])
    np.testing.assert_array_equal(np.isfinite(v), [False, False, False, True])


def test_reductions():
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=td)
    assert float(np.sum(a)) == 10.0
    assert float(np.prod(np.array([2.0, 3.0, 4.0], dtype=td))) == 24.0


def test_casts():
    x = np.array([1, 2, 3], dtype=np.int64).astype(td)
    np.testing.assert_array_equal(x.astype(np.float64), [1.0, 2.0, 3.0])
    assert np.dtype(td).itemsize == 24


def test_pickle_roundtrip():
    import pickle

    # a value with nonzero low limbs so pickling must preserve all three limbs
    a = np.array([1.0], dtype=td) + np.array([1e-18], dtype=td) + np.array([1e-30], dtype=td)
    a = np.concatenate([a, np.array([2.0, 0.5], dtype=td)])
    b = pickle.loads(pickle.dumps(a))
    assert b.dtype == np.dtype(td)
    assert bool((a == b).all())  # exact, full-precision equality


def test_registry():
    assert ud.cascade_dtypes["td_cascade"] is td
    assert "td_cascade" in ud.dtypes
