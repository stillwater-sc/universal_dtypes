"""lns (logarithmic number system) NumPy dtype family — tests (issue #9).

LNS has no ml_dtypes counterpart, so behavior is validated against Universal's own
lns semantics: values near powers of two round-trip / multiply exactly (the log
domain is exact there), add/sub are approximate (Gaussian-log), and there are
special encodings for zero and NaN but no infinity.
"""

import numpy as np
import pytest

import universal_dtypes as ud

LNS = [
    ("lns16", ud.lns16, 2),
    ("lns32", ud.lns32, 4),
]


@pytest.mark.parametrize("name,scalar,itemsize", LNS)
def test_dtype_resolves(name, scalar, itemsize):
    dt = np.dtype(scalar)
    assert dt.itemsize == itemsize
    assert np.dtype(name) == dt  # string-name resolution


@pytest.mark.parametrize("name,scalar,itemsize", LNS)
def test_roundtrip_pow2(name, scalar, itemsize):
    # powers of two (and their sums/simple ratios) are exactly representable
    vals = [1.0, 2.0, 4.0, 0.5, 0.25, -2.0, 8.0]
    a = np.array(vals, dtype=scalar)
    np.testing.assert_array_equal(a.astype(np.float64), vals)


@pytest.mark.parametrize("name,scalar,itemsize", LNS)
def test_multiply_exact_in_log_domain(name, scalar, itemsize):
    # LNS strength: multiply/divide of powers of two are exact.
    a = np.array([2.0, 4.0, 0.5, 8.0], dtype=scalar)
    b = np.array([4.0, 0.5, 8.0, 0.25], dtype=scalar)
    np.testing.assert_array_equal((a * b).astype(np.float64), [8.0, 2.0, 4.0, 2.0])
    np.testing.assert_array_equal((a / b).astype(np.float64), [0.5, 8.0, 0.0625, 32.0])


@pytest.mark.parametrize("name,scalar,itemsize", LNS)
def test_add_sub_approximate(name, scalar, itemsize):
    # add/sub go through Universal's Gaussian-log routines and are inexact — more
    # so for lns16 (rbits=8), where subtractive cancellation is coarse.
    rtol = 1e-2 if name == "lns16" else 1e-4
    atol = 5e-2 if name == "lns16" else 1e-3
    a = np.array([3.0, 10.0, 1.0], dtype=scalar)
    b = np.array([4.0, 6.0, 1.0], dtype=scalar)
    np.testing.assert_allclose((a + b).astype(np.float64), [7.0, 16.0, 2.0], rtol=rtol)
    np.testing.assert_allclose((a - b).astype(np.float64), [-1.0, 4.0, 0.0], atol=atol)


@pytest.mark.parametrize("name,scalar,itemsize", LNS)
def test_special_values(name, scalar, itemsize):
    # LNS has a zero encoding and a NaN encoding, but no infinity.
    v = np.array([0.0, np.nan, 1.0], dtype=scalar)
    np.testing.assert_array_equal(np.isnan(v), [False, True, False])
    np.testing.assert_array_equal(np.isinf(v), [False, False, False])  # LNS has no inf
    np.testing.assert_array_equal(np.isfinite(v), [True, False, True])
    out = v.astype(np.float64)
    assert out[0] == 0.0 and np.isnan(out[1]) and out[2] == 1.0


@pytest.mark.parametrize("name,scalar,itemsize", LNS)
def test_comparisons_and_sort(name, scalar, itemsize):
    a = np.array([1.0, 2.0, 4.0], dtype=scalar)
    b = np.array([1.0, 8.0, 2.0], dtype=scalar)
    np.testing.assert_array_equal(a == b, [True, False, False])
    np.testing.assert_array_equal(a < b, [False, True, False])
    c = np.array([4.0, 1.0, 2.0], dtype=scalar)
    np.testing.assert_array_equal(np.sort(c).astype(np.float64), [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(np.argsort(c), [1, 2, 0])


@pytest.mark.parametrize("name,scalar,itemsize", LNS)
def test_pickle_roundtrip(name, scalar, itemsize):
    import pickle

    a = np.array([1.0, 2.0, 4.0, 0.5], dtype=scalar)
    b = pickle.loads(pickle.dumps(a))
    assert b.dtype == np.dtype(scalar)
    np.testing.assert_array_equal(b.astype(np.float64), a.astype(np.float64))
    assert float(pickle.loads(pickle.dumps(scalar(2.0)))) == 2.0


@pytest.mark.parametrize("name,scalar,itemsize", LNS)
def test_casts_and_math(name, scalar, itemsize):
    a = np.array([1, 2, 4], dtype=np.int64).astype(scalar)
    np.testing.assert_array_equal(a.astype(np.float64), [1.0, 2.0, 4.0])
    # sqrt of perfect squares is exact in the log domain
    np.testing.assert_array_equal(
        np.sqrt(np.array([1.0, 4.0], dtype=scalar)).astype(np.float64), [1.0, 2.0]
    )


def test_lns_registry():
    assert set(ud.lns_dtypes) == {name for name, _, _ in LNS}
    for name, scalar, _ in LNS:
        assert ud.lns_dtypes[name] is scalar
    assert set(ud.lns_dtypes) <= set(ud.dtypes)
