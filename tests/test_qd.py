"""qd_cascade (quad-double) NumPy dtype — tests (issue #6).

qd_cascade is a 32-byte, ~212-bit high-precision type (unevaluated sum of four
float64) — the highest cascade tier. Same multi-word machinery as dd/td; the
distinguishing test is a four-magnitude cancellation that needs more than
triple-double precision to retain.
"""

from decimal import Decimal, getcontext

import numpy as np

import universal_dtypes as ud

qd = ud.qd_cascade
getcontext().prec = 70  # beyond qd's ~62 decimal digits


def test_dtype_resolves():
    dt = np.dtype(qd)
    assert dt.itemsize == 32  # four float64 limbs
    assert np.dtype("qd_cascade") == dt


def test_array_creation_and_exact_roundtrip():
    vals = [1.0, 2.0, 3.0, 0.5, -2.5, 1e20, 1e-20]
    a = np.array(vals, dtype=qd)
    np.testing.assert_array_equal(a.astype(np.float64), vals)


def test_four_magnitude_cancellation():
    # 1e30 + 1 + 1e-15 + 1e-30 - 1e30 - 1 - 1e-15 == 1e-30 exactly. Spanning four
    # magnitudes (1e30 .. 1e-30, ratio 1e60 ~ 2^199) needs ~212-bit intermediate
    # precision — triple-double is not enough; quad-double retains it.
    parts = [1e30, 1.0, 1e-15, 1e-30, -1e30, -1.0, -1e-15]
    acc = np.array([parts[0]], dtype=qd)
    for p in parts[1:]:
        acc = acc + np.array([p], dtype=qd)
    assert float(acc[0]) == 1e-30
    # float64 collapses the whole thing away from the true 1e-30
    f64 = 0.0
    for p in parts:
        f64 += p
    assert abs(f64 - 1e-30) > 0.5


def test_full_precision_comparison():
    one = np.array([1.0], dtype=qd)
    bigger = one + np.array([1e-40], dtype=qd)  # below dd and td range
    assert float(one[0]) == float(bigger[0])
    assert bool((one < bigger)[0])
    assert not bool((one == bigger)[0])


def test_sort_below_double_precision():
    base = np.array([1.0, 1.0, 1.0], dtype=qd)
    arr = base + np.array([2e-40, 0.0, 1e-40], dtype=qd)
    np.testing.assert_array_equal(np.argsort(arr), [1, 2, 0])


def test_arithmetic_matches_decimal_oracle():
    a_f, b_f = 1.1, 1.3
    prod = np.array([a_f], dtype=qd) * np.array([b_f], dtype=qd)
    got = float(prod[0])
    ref = float(Decimal(a_f) * Decimal(b_f))
    assert abs(got - ref) < 1e-15


def test_special_values():
    v = np.array([np.inf, -np.inf, np.nan, 3.0], dtype=qd)
    np.testing.assert_array_equal(np.isinf(v), [True, True, False, False])
    np.testing.assert_array_equal(np.isnan(v), [False, False, True, False])
    np.testing.assert_array_equal(np.isfinite(v), [False, False, False, True])


def test_reductions():
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=qd)
    assert float(np.sum(a)) == 10.0
    assert float(np.prod(np.array([2.0, 3.0, 4.0], dtype=qd))) == 24.0


def test_casts():
    x = np.array([1, 2, 3], dtype=np.int64).astype(qd)
    np.testing.assert_array_equal(x.astype(np.float64), [1.0, 2.0, 3.0])
    assert np.dtype(qd).itemsize == 32


def test_pickle_roundtrip():
    import pickle

    # a value with nonzero limbs across all four words
    a = np.array([1.0], dtype=qd)
    for e in (1e-16, 1e-32, 1e-48):
        a = a + np.array([e], dtype=qd)
    a = np.concatenate([a, np.array([2.0, 0.5], dtype=qd)])
    b = pickle.loads(pickle.dumps(a))
    assert b.dtype == np.dtype(qd)
    assert bool((a == b).all())  # exact, full-precision equality


def test_registry():
    assert ud.cascade_dtypes["qd_cascade"] is qd
    assert "qd_cascade" in ud.dtypes
    # the cascade family is now complete
    assert set(ud.cascade_dtypes) == {"dd_cascade", "td_cascade", "qd_cascade"}
